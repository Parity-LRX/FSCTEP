from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from molecular_force_field.cli.export_libtorch_core import export_core
from molecular_force_field.interfaces.lammps_mliap import _recover_cartesian_physical_tensors
from molecular_force_field.models import (
    PureCartesianSparseTransformerLayer,
    PureCartesianSparseTransformerLayerSave,
)
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.utils.external_tensor_specs import pack_external_tensor_dict
from molecular_force_field.utils.tensor_utils import derive_born_effective_charge_from_forces


def _cartesian_kwargs(device: torch.device) -> dict:
    cfg = ModelConfig(dtype=torch.float64)
    return dict(
        max_embed_radius=cfg.max_radius,
        main_max_radius=cfg.max_radius_main,
        main_number_of_basis=cfg.number_of_basis_main,
        hidden_dim_conv=cfg.channel_in,
        hidden_dim_sh=cfg.get_hidden_dim_sh(),
        hidden_dim=cfg.emb_number_main_2,
        channel_in2=cfg.channel_in2,
        embedding_dim=cfg.embedding_dim,
        max_atomvalue=cfg.max_atomvalue,
        output_size=cfg.output_size,
        embed_size=cfg.embed_size,
        main_hidden_sizes3=cfg.main_hidden_sizes3,
        num_layers=cfg.num_layers,
        num_interaction=2,
        invariant_channels=32,
        function_type_main=cfg.function_type,
        lmax=cfg.lmax,
        max_rank_other=1,
        k_policy="k0",
        device=device,
    )


def _toy_batch(device: torch.device, dtype: torch.dtype):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.8, 0.2], [1.1, 0.7, 0.3]],
        dtype=dtype,
        device=device,
    )
    A = torch.tensor([1, 8, 1, 8], dtype=torch.long, device=device)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    edge_src = torch.tensor([0, 1, 1, 0, 2, 3, 3, 2], dtype=torch.long, device=device)
    edge_dst = torch.tensor([1, 0, 0, 1, 3, 2, 2, 3], dtype=torch.long, device=device)
    edge_shifts = torch.zeros(edge_src.shape[0], 3, dtype=dtype, device=device)
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(2, 1, 1) * 20.0
    return pos, A, batch, edge_src, edge_dst, edge_shifts, cell


def _make_sparse_checkpoint(
    path: str,
    *,
    mode: str,
    external_tensor_rank: int | None,
    physical_tensor_outputs: dict[str, dict] | None,
) -> None:
    device = torch.device("cpu")
    cfg = ModelConfig(dtype=torch.float64)
    cls = (
        PureCartesianSparseTransformerLayerSave
        if mode == "pure-cartesian-sparse-save"
        else PureCartesianSparseTransformerLayer
    )
    model = cls(
        **_cartesian_kwargs(device),
        external_tensor_rank=external_tensor_rank,
        physical_tensor_outputs=physical_tensor_outputs,
    ).to(device)
    ckpt = {
        "e3trans_state_dict": model.state_dict(),
        "dtype": "float64",
        "tensor_product_mode": mode,
        "atomic_energy_keys": torch.tensor([1, 8], dtype=torch.long),
        "atomic_energy_values": torch.tensor([-13.6, -75.0], dtype=cfg.dtype),
        "max_radius": cfg.max_radius,
        "model_hyperparameters": {
            "num_interaction": 2,
            "invariant_channels": 32,
            "max_rank_other": 1,
            "k_policy": "k0",
            "external_tensor_rank": external_tensor_rank,
            "physical_tensor_outputs": physical_tensor_outputs,
        },
    }
    if external_tensor_rank is not None:
        ckpt["external_tensor_rank"] = external_tensor_rank
    if physical_tensor_outputs is not None:
        ckpt["physical_tensor_outputs"] = physical_tensor_outputs
    torch.save(ckpt, path)


def _run_export_forward(core: torch.nn.Module, *, external_tensor_rank: int | None):
    device = torch.device("cpu")
    dtype = torch.float32
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _toy_batch(device, dtype)
    batch = torch.zeros_like(batch)
    cell = cell[:1]
    edge_vec = (pos[edge_dst] - pos[edge_src]).detach()
    if external_tensor_rank is None:
        external_tensor = torch.empty(0, dtype=dtype, device=device)
    else:
        external_tensor = torch.zeros((3,) * external_tensor_rank, dtype=dtype, device=device)
    with torch.no_grad():
        return core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)


def test_sparse_forward_supports_external_tensor_and_physical_outputs() -> None:
    physical_tensor_outputs = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
        "born_effective_charge_per_atom": {"ls": [0, 1, 2], "channels_out": 1, "reduce": "none"},
    }
    for cls in (PureCartesianSparseTransformerLayer, PureCartesianSparseTransformerLayerSave):
        model = cls(
            **_cartesian_kwargs(torch.device("cpu")),
            external_tensor_rank=1,
            physical_tensor_outputs=physical_tensor_outputs,
        ).to(dtype=torch.float64)
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _toy_batch(torch.device("cpu"), torch.float64)
        atom_energy, physical_out = model(
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            external_tensor=torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
            return_physical_tensors=True,
        )
        assert atom_energy.shape == (4, 1)
        assert physical_out["dipole"][1].shape == (2, 1, 3)
        assert physical_out["dipole_per_atom"][1].shape == (4, 1, 3)
        assert physical_out["polarizability"][0].shape == (2, 1, 1)
        assert physical_out["polarizability"][2].shape == (2, 1, 3, 3)
        assert physical_out["polarizability_per_atom"][2].shape == (4, 1, 3, 3)
        assert physical_out["born_effective_charge_per_atom"][0].shape == (4, 1, 1)
        assert physical_out["born_effective_charge_per_atom"][1].shape == (4, 1, 3)
        assert physical_out["born_effective_charge_per_atom"][2].shape == (4, 1, 3, 3)
        global_phys, atom_phys, global_mask, atom_mask = _recover_cartesian_physical_tensors(
            physical_out,
            num_graphs=2,
            num_nodes=4,
            device=torch.device("cpu"),
            dtype=atom_energy.dtype,
        )
        assert global_phys.shape == (2, 22)
        assert atom_phys.shape == (4, 31)
        assert global_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
        assert atom_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]


def test_derive_bec_from_forces_matches_analytic_jacobian() -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    batch = torch.tensor([0, 0, 1], dtype=torch.long, device=device)
    external_field = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    coeff = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [4.0, 0.0, -2.0]],
            [[-1.0, 0.2, 1.5], [2.5, 3.0, -0.5], [0.0, 1.0, 1.0]],
            [[0.7, -0.3, 0.9], [1.1, 0.0, -2.2], [3.3, -1.1, 0.4]],
        ],
        dtype=dtype,
        device=device,
    )
    forces = torch.einsum("nab,nb->na", coeff, external_field[batch])
    bec = derive_born_effective_charge_from_forces(
        forces,
        external_field,
        batch,
        create_graph=False,
    )
    assert torch.allclose(bec, coeff)


def test_sparse_return_physical_without_heads_raises() -> None:
    model = PureCartesianSparseTransformerLayer(
        **_cartesian_kwargs(torch.device("cpu")),
    ).to(dtype=torch.float64)
    pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _toy_batch(torch.device("cpu"), torch.float64)
    try:
        model(
            pos,
            A,
            batch,
            edge_src,
            edge_dst,
            edge_shifts,
            cell,
            return_physical_tensors=True,
        )
    except ValueError as exc:
        assert "physical_tensor_outputs was not set" in str(exc)
    else:
        raise AssertionError("Expected ValueError when requesting physical tensors without heads")


def test_sparse_forward_supports_simultaneous_rank1_fields() -> None:
    specs = [
        {"name": "external_field", "rank": 1, "irrep": "1o"},
        {"name": "magnetic_field", "rank": 1, "irrep": "1e"},
    ]
    for cls in (PureCartesianSparseTransformerLayer, PureCartesianSparseTransformerLayerSave):
        model = cls(
            **_cartesian_kwargs(torch.device("cpu")),
            external_tensor_specs=specs,
            physical_tensor_outputs={
                "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
            },
        ).to(dtype=torch.float64)
        pos, A, batch, edge_src, edge_dst, edge_shifts, cell = _toy_batch(torch.device("cpu"), torch.float64)
        packed = pack_external_tensor_dict(
            {
                "external_field": torch.tensor([0.1, 0.0, -0.2], dtype=torch.float64),
                "magnetic_field": torch.tensor([-0.3, 0.2, 0.4], dtype=torch.float64),
            },
            specs,
            dtype=torch.float64,
        )
        atom_energy, physical_out = model(
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
        assert atom_energy.shape == (4, 1)
        assert physical_out["dipole"][1].shape == (2, 1, 3)


def test_sparse_export_core_physical_tensor_tuple_schema() -> None:
    physical_tensor_outputs = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
        "born_effective_charge_per_atom": {"ls": [0, 1, 2], "channels_out": 1, "reduce": "none"},
    }
    for mode in ("pure-cartesian-sparse", "pure-cartesian-sparse-save"):
        with tempfile.TemporaryDirectory(prefix="mff-sparse-export-") as td:
            tmpdir = Path(td)
            ckpt = tmpdir / f"{mode}.pth"
            out_pt = tmpdir / "core.pt"
            e0_csv = tmpdir / "e0.csv"
            _make_sparse_checkpoint(
                str(ckpt),
                mode=mode,
                external_tensor_rank=1,
                physical_tensor_outputs=physical_tensor_outputs,
            )
            e0_csv.write_text("Atom,E0\n1,-13.6\n8,-75.0\n", encoding="utf-8")
            export_core(
                checkpoint=str(ckpt),
                elements=["H", "O"],
                device="cpu",
                max_radius=5.0,
                num_interaction=2,
                out_pt=str(out_pt),
                tensor_product_mode=mode,
                embed_e0=True,
                e0_csv=str(e0_csv),
            )
            meta = json.loads(out_pt.with_suffix(".pt.json").read_text(encoding="utf-8"))
            assert meta["external_tensor_rank"] == 1
            core = torch.jit.load(str(out_pt), map_location="cpu")
            core.eval()
            out = _run_export_forward(core, external_tensor_rank=1)
            assert isinstance(out, tuple)
            assert len(out) == 6
            atom_energy, global_phys, atom_phys, global_mask, atom_mask, reciprocal_source = out
            assert atom_energy.shape == (4, 1)
            assert global_phys.shape == (1, 22)
            assert atom_phys.shape == (4, 31)
            assert global_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
            assert atom_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 1.0]
            assert reciprocal_source.shape == (4, 0)
