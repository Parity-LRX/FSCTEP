from __future__ import annotations

import torch

from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.models.pure_cartesian_ictd_layers_full import (
    PureCartesianICTDTransformerLayer as PureCartesianICTDTransformerLayerFull,
)
from molecular_force_field.models.pure_cartesian_ictd_layers_o3 import PureCartesianICTDO3TransformerLayer
from molecular_force_field.models.pure_cartesian_sparse_layers import PureCartesianSparseTransformerLayer


def _toy_graph_inputs(dtype: torch.dtype = torch.float64):
    pos = torch.randn(4, 3, dtype=dtype)
    A = torch.tensor([1, 1, 8, 1], dtype=torch.long)
    batch = torch.zeros(4, dtype=torch.long)
    edge_src = torch.tensor([0, 0, 1, 2, 3, 1], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 2, 3, 0, 3], dtype=torch.long)
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype)
    cell = torch.eye(3, dtype=dtype).unsqueeze(0)
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _set_fidelity_embedding(conv, level1_value: float = 0.5) -> None:
    with torch.no_grad():
        conv.fidelity_embedding.weight.zero_()
        conv.fidelity_embedding.weight[1].fill_(level1_value)


def _configure_delta_branch(model, fid: int, amount: float) -> None:
    with torch.no_grad():
        head = model.delta_proj_total[str(fid)]
        head.output.weight.zero_()
        head.output.bias.zero_()
        head.output.bias[0] = float(amount)
        sum_head = model.delta_weighted_sum[str(fid)]
        sum_head.weights.zero_()
        sum_head.weights[0] = 1.0


def test_collate_h5_batches_graph_level_fidelity_ids() -> None:
    sample0 = {
        "pos": torch.zeros(2, 3, dtype=torch.float64),
        "A": torch.tensor([1, 8], dtype=torch.long),
        "y": torch.tensor([0.0], dtype=torch.float64),
        "force": torch.zeros(2, 3, dtype=torch.float64),
        "edge_src": torch.tensor([0], dtype=torch.long),
        "edge_dst": torch.tensor([1], dtype=torch.long),
        "edge_shifts": torch.zeros(1, 3, dtype=torch.float64),
        "cell": torch.eye(3, dtype=torch.float64),
        "stress": torch.zeros(3, 3, dtype=torch.float64),
        "fidelity_id": torch.tensor(0, dtype=torch.long),
    }
    sample1 = {
        **sample0,
        "fidelity_id": torch.tensor(2, dtype=torch.long),
    }
    batch = collate_fn_h5([sample0, sample1])
    extras = batch[-1]
    assert "fidelity_id" in extras
    assert extras["fidelity_id"].dtype == torch.long
    assert extras["fidelity_id"].tolist() == [0, 2]


def test_ictd_multifidelity_changes_prediction() -> None:
    dtype = torch.float64
    model = PureCartesianICTDTransformerLayerFull(
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
        num_fidelity_levels=2,
    ).to(dtype=dtype)
    _set_fidelity_embedding(model.e3_conv_emb)
    inputs = _toy_graph_inputs(dtype)
    out0 = model.e3_conv_emb(*inputs, fidelity_ids=torch.tensor([0], dtype=torch.long))
    out1 = model.e3_conv_emb(*inputs, fidelity_ids=torch.tensor([1], dtype=torch.long))
    assert not torch.allclose(out0, out1)


def test_o3_multifidelity_changes_prediction() -> None:
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
        num_fidelity_levels=2,
        o3_irrep_preset="minimal",
    ).to(dtype=dtype)
    _set_fidelity_embedding(model.e3_conv_emb)
    inputs = _toy_graph_inputs(dtype)
    out0 = model.e3_conv_emb(*inputs, fidelity_ids=torch.tensor([0], dtype=torch.long))
    out1 = model.e3_conv_emb(*inputs, fidelity_ids=torch.tensor([1], dtype=torch.long))
    assert not torch.allclose(out0, out1)


def test_sparse_multifidelity_changes_prediction() -> None:
    dtype = torch.float64
    model = PureCartesianSparseTransformerLayer(
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
        num_fidelity_levels=2,
    ).to(dtype=dtype)
    _set_fidelity_embedding(model.e3_conv_emb)
    inputs = _toy_graph_inputs(dtype)
    out0 = model.e3_conv_emb(*inputs, fidelity_ids=torch.tensor([0], dtype=torch.long))
    out1 = model.e3_conv_emb(*inputs, fidelity_ids=torch.tensor([1], dtype=torch.long))
    assert not torch.allclose(out0, out1)


def _assert_delta_baseline_behavior(model) -> None:
    dtype = torch.float64
    model = model.to(dtype=dtype)
    inputs = _toy_graph_inputs(dtype)
    fid0 = torch.tensor([0], dtype=torch.long)
    fid1 = torch.tensor([1], dtype=torch.long)
    with torch.no_grad():
        base0 = model(*inputs, fidelity_ids=fid0)
        base1 = model(*inputs, fidelity_ids=fid1)
    assert torch.allclose(base0, base1), "delta heads should start at zero output"
    _configure_delta_branch(model, fid=1, amount=0.25)
    with torch.no_grad():
        out0 = model(*inputs, fidelity_ids=fid0)
        out1 = model(*inputs, fidelity_ids=fid1)
    assert torch.allclose(out0, base0), "baseline branch should remain unchanged"
    assert not torch.allclose(out1, base1), "fidelity-1 branch should include delta correction"


def test_ictd_delta_baseline_branching() -> None:
    model = PureCartesianICTDTransformerLayerFull(
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
        num_fidelity_levels=2,
        multi_fidelity_mode="delta-baseline",
    )
    _assert_delta_baseline_behavior(model)


def test_o3_delta_baseline_branching() -> None:
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
        num_fidelity_levels=2,
        multi_fidelity_mode="delta-baseline",
        o3_irrep_preset="minimal",
    )
    _assert_delta_baseline_behavior(model)


def test_sparse_delta_baseline_branching() -> None:
    model = PureCartesianSparseTransformerLayer(
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
        num_fidelity_levels=2,
        multi_fidelity_mode="delta-baseline",
    )
    _assert_delta_baseline_behavior(model)
