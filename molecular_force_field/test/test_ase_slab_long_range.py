from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read as ase_read, write as ase_write

from molecular_force_field.active_learning.labeling import IdentityLabeler
from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
)
from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu


def test_radius_graph_respects_partial_pbc_slab() -> None:
    pos = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 18.0]], dtype=torch.float64)
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]],
        dtype=torch.float64,
    ).unsqueeze(0)

    edge_src_slab, edge_dst_slab, edge_shifts_slab = radius_graph_pbc_gpu(
        pos,
        5.0,
        cell,
        pbc=(True, True, False),
    )
    assert edge_src_slab.numel() == 0
    assert edge_dst_slab.numel() == 0
    assert edge_shifts_slab.numel() == 0

    edge_src_periodic, edge_dst_periodic, edge_shifts_periodic = radius_graph_pbc_gpu(
        pos,
        5.0,
        cell,
        pbc=(True, True, True),
    )
    assert edge_src_periodic.numel() == 2
    assert edge_dst_periodic.numel() == 2
    assert edge_shifts_periodic.tolist() == [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]


def test_identity_labeler_runs_with_slab_long_range_checkpoint() -> None:
    with tempfile.TemporaryDirectory(prefix="mff-ase-slab-test-") as td:
        tmpdir = Path(td)
        ckpt = tmpdir / "slab_long_range_mesh.pth"
        input_xyz = tmpdir / "input.xyz"
        output_xyz = tmpdir / "output.xyz"

        _make_dummy_checkpoint_pure_cartesian_ictd(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="reciprocal-spectral-v1",
            long_range_boundary="slab",
            long_range_hidden_dim=32,
            long_range_filter_hidden_dim=16,
            long_range_mesh_size=8,
            long_range_slab_padding_factor=2,
            long_range_source_channels=1,
            long_range_reciprocal_backend="mesh_fft",
            long_range_energy_partition="potential",
            long_range_green_mode="poisson",
            long_range_assignment="cic",
        )

        atoms = Atoms(
            symbols=["H", "H"],
            positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 18.0]],
            cell=[10.0, 10.0, 20.0],
            pbc=[True, True, False],
        )
        ase_write(input_xyz, [atoms], format="extxyz")

        labeler = IdentityLabeler(str(ckpt), device="cpu", max_radius=5.0)
        labeler.label(str(input_xyz), str(output_xyz), str(tmpdir))

        text = output_xyz.read_text(encoding="utf-8")
        assert 'pbc="T T F"' in text
        assert "energy=" in text
        assert "forces:R:3" in text

        labeled = ase_read(str(output_xyz), index=":")
        assert len(labeled) == 1
        assert labeled[0].get_pbc().tolist() == [True, True, False]
