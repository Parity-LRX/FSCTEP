from __future__ import annotations

import torch

from molecular_force_field.models.pure_cartesian_ictd_layers import (
    PhysicalTensorICTDEmbedding,
    PhysicalTensorICTDRecovery,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer,
)


def test_full_rank2_ictd_roundtrip_recovers_general_matrix() -> None:
    x = torch.tensor(
        [
            [[2.0, 0.7, -0.4], [0.1, 1.5, 0.8], [0.3, -0.2, -0.2]],
            [[-1.0, 0.6, 0.2], [-0.4, 0.5, -0.6], [0.9, -0.3, 0.7]],
        ],
        dtype=torch.float64,
    )
    embed = PhysicalTensorICTDEmbedding(
        rank=2,
        lmax_out=2,
        channels_in=1,
        channels_out=1,
        input_repr="cartesian",
        include_trace_chain=True,
        rank2_mode="full",
    ).to(dtype=torch.float64)
    recover = PhysicalTensorICTDRecovery(
        rank=2,
        channels_in=1,
        lmax_in=2,
        include_trace_chain=True,
        rank2_mode="full",
    ).to(dtype=torch.float64)

    blocks = embed(x, return_blocks=True)
    assert set(blocks.keys()) >= {0, 1, 2}
    x_rec = recover(blocks)
    assert torch.allclose(x_rec, x, atol=1e-6, rtol=1e-6)


def test_ictd_physical_tensor_spec_accepts_irreps_schema() -> None:
    model = PureCartesianICTDTransformerLayer(
        max_embed_radius=4.0,
        main_max_radius=4.0,
        main_number_of_basis=4,
        hidden_dim_conv=8,
        hidden_dim_sh=8,
        hidden_dim=8,
        channel_in2=8,
        embedding_dim=8,
        max_atomvalue=4,
        output_size=4,
        embed_size=[8, 8],
        main_hidden_sizes3=[8, 8],
        num_layers=1,
        num_interaction=2,
        lmax=2,
        physical_tensor_outputs={
            "magnetic_moment": {"irreps": ["1e"], "channels_out": 1, "reduce": "sum"},
            "born_effective_charge_per_atom": {
                "irreps": ["0e", "1e", "2e"],
                "channels_out": 1,
                "reduce": "none",
            },
        },
        external_tensor_rank=1,
        external_tensor_irrep="1e",
    )
    assert model.external_tensor_irrep == "1e"
    spec = model._physical_tensor_specs
    assert spec is not None
    assert spec["magnetic_moment"]["irreps"] == ["1e"]
    assert spec["born_effective_charge_per_atom"]["irreps"] == ["0e", "1e", "2e"]
    assert spec["born_effective_charge_per_atom"]["ls"] == [0, 1, 2]
