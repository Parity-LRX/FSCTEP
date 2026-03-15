from __future__ import annotations

import torch

from molecular_force_field.models.ictd_irreps import HarmonicFullyConnectedTensorProductO3
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import (
    PureCartesianICTDO3TransformerLayer,
    resolve_o3_active_irreps,
)
from molecular_force_field.utils.external_tensor_specs import pack_external_tensor_dict


def _make_o3_blocks(batch: int = 2, mul: int = 1, dtype: torch.dtype = torch.float64):
    active = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    blocks = {}
    for l, p in active:
        blocks[(l, p)] = torch.zeros(batch, mul, 2 * l + 1, dtype=dtype)
    return active, blocks


def test_o3_tp_respects_parity_multiplication_rule() -> None:
    active, x1 = _make_o3_blocks()
    _, x2 = _make_o3_blocks()
    x1[(1, 1)][..., 0] = 1.0
    x2[(1, -1)][..., 0] = 1.0

    tp = HarmonicFullyConnectedTensorProductO3(
        mul_in1=1,
        mul_in2=1,
        mul_out=1,
        lmax=2,
        active_irreps=active,
        internal_weights=True,
    )
    weights = torch.ones(x1[(0, 1)].shape[0], tp.num_paths, dtype=torch.float64)
    out = tp(x1, x2, weights)

    nonzero_even = sum(v.abs().sum().item() for (l, p), v in out.items() if p == 1)
    nonzero_odd = sum(v.abs().sum().item() for (l, p), v in out.items() if p == -1)
    assert nonzero_odd > 0.0
    assert nonzero_even == 0.0


def test_o3_model_forward_external_and_physical_heads() -> None:
    dtype = torch.float64
    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=10,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        external_tensor_rank=1,
        external_tensor_irrep="1e",
        physical_tensor_outputs={
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
            "born_effective_charge_per_atom": {"irreps": ["0e", "1e", "2e"], "channels_out": 1, "reduce": "none"},
        },
    ).to(dtype=dtype)

    pos = torch.randn(4, 3, dtype=dtype)
    A = torch.tensor([1, 1, 8, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    edge_src = torch.tensor([0, 0, 1, 2, 3, 1], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 2, 3, 0, 3], dtype=torch.long)
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    out_e, phys = model(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        external_tensor=torch.tensor([0.0, 0.0, 0.1], dtype=dtype),
        return_physical_tensors=True,
    )
    assert out_e.shape == (4, 1)
    assert phys["magnetic_moment"][1].shape == (1, 1, 3)
    assert phys["born_effective_charge_per_atom"][0].shape == (4, 1, 1)
    assert phys["born_effective_charge_per_atom"][1].shape == (4, 1, 3)
    assert phys["born_effective_charge_per_atom"][2].shape == (4, 1, 5)


def test_o3_active_irrep_presets_and_explicit_override() -> None:
    phys = {
        "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
        "born_effective_charge_per_atom": {"irreps": ["0e", "1e", "2e"], "channels_out": 1, "reduce": "none"},
    }
    auto_irreps = resolve_o3_active_irreps(
        lmax=2,
        preset="auto",
        external_tensor_irrep="1e",
        physical_tensor_outputs=phys,
    )
    minimal_irreps = resolve_o3_active_irreps(
        lmax=2,
        preset="minimal",
        external_tensor_irrep="1e",
        physical_tensor_outputs=phys,
    )
    explicit_irreps = resolve_o3_active_irreps(
        lmax=2,
        preset="full",
        explicit_irreps="0e,1e",
        external_tensor_irrep="1e",
        physical_tensor_outputs=phys,
    )

    assert auto_irreps == [(0, 1), (1, 1), (1, -1), (2, 1)]
    assert minimal_irreps == [(0, 1), (1, 1), (2, 1)]
    assert explicit_irreps == [(0, 1), (1, 1), (2, 1)]


def test_o3_model_minimal_preset_tracks_required_irreps() -> None:
    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=10,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        external_tensor_rank=1,
        external_tensor_irrep="1e",
        physical_tensor_outputs={
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
        },
        o3_irrep_preset="minimal",
    ).to(dtype=torch.float64)

    assert model.active_irreps == [(0, 1), (1, 1)]


def test_o3_model_supports_simultaneous_electric_and_magnetic_fields() -> None:
    dtype = torch.float64
    specs = [
        {"name": "external_field", "rank": 1, "irrep": "1o"},
        {"name": "magnetic_field", "rank": 1, "irrep": "1e"},
    ]
    model = PureCartesianICTDO3TransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=10,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        external_tensor_specs=specs,
        physical_tensor_outputs={
            "dipole": {"irreps": ["1o"], "channels_out": 1, "reduce": "sum"},
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
        },
        o3_irrep_preset="auto",
    ).to(dtype=dtype)

    pos = torch.randn(4, 3, dtype=dtype)
    A = torch.tensor([1, 1, 8, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    edge_src = torch.tensor([0, 0, 1, 2, 3, 1], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 2, 3, 0, 3], dtype=torch.long)
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    packed = pack_external_tensor_dict(
        {
            "external_field": torch.tensor([0.1, -0.2, 0.3], dtype=dtype),
            "magnetic_field": torch.tensor([-0.3, 0.4, 0.2], dtype=dtype),
        },
        specs,
        dtype=dtype,
    )
    out_e, phys = model(
        pos,
        A,
        batch,
        edge_src,
        edge_dst,
        edge_shifts,
        cell,
        external_tensor=packed,
        return_physical_tensors=True,
    )

    assert packed is not None
    assert packed.shape == (6,)
    assert out_e.shape == (4, 1)
    assert phys["dipole"][1].shape == (1, 1, 3)
    assert phys["magnetic_moment"][1].shape == (1, 1, 3)
