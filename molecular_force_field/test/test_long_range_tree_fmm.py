from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from molecular_force_field.cli.export_libtorch_core import export_core
from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
)
from molecular_force_field.models.long_range import LatentCoulombLongRange


def test_tree_fmm_matches_dense_pairwise_when_theta_zero() -> None:
    torch.manual_seed(0)
    feature_dim = 6
    hidden_dim = 8
    invariant_features = torch.randn(6, feature_dim, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1) * 20.0

    dense = LatentCoulombLongRange(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        boundary="nonperiodic",
        backend="dense_pairwise",
        energy_partition="potential",
    ).to(dtype=torch.float64)
    tree = LatentCoulombLongRange(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        boundary="nonperiodic",
        backend="tree_fmm",
        energy_partition="potential",
        theta=0.0,
        leaf_size=1,
        multipole_order=0,
    ).to(dtype=torch.float64)
    tree.load_state_dict(dense.state_dict(), strict=True)

    with torch.no_grad():
        dense.energy_scale.fill_(1.0)
        tree.energy_scale.fill_(1.0)

    pos_dense = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [1.0, 0.5, 0.7],
            [0.2, 1.3, 0.4],
            [4.0, 4.1, 4.2],
            [4.8, 4.3, 4.4],
            [4.2, 4.9, 4.1],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    pos_tree = pos_dense.detach().clone().requires_grad_(True)

    atom_energy_dense = dense(invariant_features, pos_dense, batch, cell)
    atom_energy_tree = tree(invariant_features, pos_tree, batch, cell)
    total_dense = atom_energy_dense.sum()
    total_tree = atom_energy_tree.sum()
    forces_dense = -torch.autograd.grad(total_dense, pos_dense)[0]
    forces_tree = -torch.autograd.grad(total_tree, pos_tree)[0]

    torch.testing.assert_close(atom_energy_tree, atom_energy_dense, atol=1.0e-8, rtol=1.0e-6)
    torch.testing.assert_close(forces_tree, forces_dense, atol=1.0e-8, rtol=1.0e-6)


def test_export_core_writes_tree_fmm_runtime_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="mff-tree-export-") as td:
        tmpdir = Path(td)
        ckpt = tmpdir / "tree_fmm_latent.pth"
        out_pt = tmpdir / "core.pt"
        _make_dummy_checkpoint_pure_cartesian_ictd(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="latent-coulomb",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_backend="tree_fmm",
            long_range_energy_partition="potential",
            long_range_theta=0.4,
            long_range_leaf_size=5,
            long_range_multipole_order=0,
        )

        export_core(
            checkpoint=str(ckpt),
            elements=["H", "O"],
            device="cpu",
            max_radius=None,
            num_interaction=None,
            out_pt=str(out_pt),
            tensor_product_mode="pure-cartesian-ictd",
            embed_e0=False,
            export_reciprocal_source=True,
        )

        meta = json.loads(out_pt.with_suffix(".pt.json").read_text(encoding="utf-8"))
        assert meta["export_reciprocal_source"] is True
        assert meta["long_range_runtime_backend"] == "tree_fmm"
        assert meta["long_range_source_kind"] == "latent_charge"
        assert meta["long_range_source_channels"] == 1
        assert meta["long_range_source_layout"] == "scalar_per_atom"
        assert meta["long_range_boundary"] == "nonperiodic"
        assert meta["long_range_backend"] == "tree_fmm"
        assert meta["long_range_energy_partition"] == "potential"
        assert meta["long_range_theta"] == 0.4
        assert meta["long_range_leaf_size"] == 5
        assert meta["long_range_multipole_order"] == 0
        assert meta["long_range_screening"] is not None
        assert meta["long_range_softening"] is not None
        assert meta["long_range_energy_scale"] is not None
