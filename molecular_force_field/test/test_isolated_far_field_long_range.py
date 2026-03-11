from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from molecular_force_field.cli.export_libtorch_core import export_core
from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
    _make_dummy_checkpoint_spherical_save_cue,
)
from molecular_force_field.models.long_range import build_long_range_module


def _manual_shell_stats(
    module,
    pos: torch.Tensor,
    source: torch.Tensor,
    *,
    cutoff: float,
    shell_bounds: list[tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_atoms = pos.size(0)
    source_dim = source.size(1)
    out = source.new_zeros((n_atoms, len(shell_bounds), source_dim + 3))
    explicit_sum = source.new_zeros((n_atoms, source_dim))
    explicit_count = source.new_zeros((n_atoms, 1))
    disp = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist = torch.linalg.vector_norm(disp, dim=-1)
    pair_mask = (~torch.eye(n_atoms, dtype=torch.bool, device=pos.device)) & (dist > cutoff)
    explicit_max = shell_bounds[-1][1]
    explicit_mask = pair_mask & (dist <= explicit_max)
    for dst in range(n_atoms):
        src_idx = torch.nonzero(explicit_mask[dst], as_tuple=False).view(-1)
        if src_idx.numel() > 0:
            explicit_sum[dst] = source.index_select(0, src_idx).sum(dim=0)
            explicit_count[dst, 0] = float(src_idx.numel())
        for shell_idx, (lower, upper) in enumerate(shell_bounds):
            shell_src = torch.nonzero(pair_mask[dst] & (dist[dst] <= upper) & (dist[dst] > lower), as_tuple=False).view(-1)
            if shell_src.numel() == 0:
                continue
            shell_source = source.index_select(0, shell_src)
            shell_dist = dist[dst].index_select(0, shell_src).unsqueeze(-1)
            shell_weight = module._physical_kernel(shell_dist.squeeze(-1)).unsqueeze(-1).to(dtype=source.dtype)
            weight_sum = shell_weight.sum(dim=0, keepdim=True)
            out[dst, shell_idx, :source_dim] = (shell_source * shell_weight).sum(dim=0) / weight_sum.squeeze(0)
            out[dst, shell_idx, source_dim:source_dim + 1] = weight_sum
            out[dst, shell_idx, source_dim + 1:source_dim + 2] = (shell_dist * shell_weight).sum(dim=0) / weight_sum.squeeze(0)
            out[dst, shell_idx, source_dim + 2:source_dim + 3] = torch.sqrt(
                (shell_dist.square() * shell_weight).sum(dim=0) / weight_sum.squeeze(0) + 1.0e-8
            )
    return out, explicit_sum, explicit_count


def test_isolated_far_field_forward_shape() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v1",
        feature_dim=12,
        hidden_dim=16,
        boundary="nonperiodic",
        neutralize=True,
        source_channels=8,
        energy_partition="potential",
    ).to(dtype=torch.float64)

    x = torch.randn(5, 12, dtype=torch.float64)
    pos = torch.randn(5, 3, dtype=torch.float64)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    edge_src = torch.tensor([0, 1, 3], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 4], dtype=torch.long)

    out = module(x, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst)
    assert out.shape == (5, 1)


def test_isolated_far_field_v2_forward_shape() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v2",
        feature_dim=12,
        hidden_dim=16,
        boundary="nonperiodic",
        neutralize=True,
        energy_partition="potential",
        far_source_dim=6,
        far_num_shells=2,
        far_shell_growth=2.0,
        far_tail=True,
        far_stats="mean,count,mean_r,rms_r",
        far_source_norm=True,
        far_gate_init=0.0,
        cutoff_radius=1.5,
    ).to(dtype=torch.float64)

    x = torch.randn(5, 12, dtype=torch.float64)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(2, 1, 1)
    edge_src = torch.tensor([0, 1], dtype=torch.long)
    edge_dst = torch.tensor([1, 0], dtype=torch.long)

    out = module(x, pos, batch, cell, edge_src=edge_src, edge_dst=edge_dst)
    assert out.shape == (5, 1)


def test_isolated_far_field_v2_sparse_shell_stats_match_dense_reference() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v2",
        feature_dim=6,
        hidden_dim=8,
        boundary="nonperiodic",
        neutralize=False,
        energy_partition="potential",
        far_source_dim=3,
        far_num_shells=2,
        far_shell_growth=2.0,
        far_tail=False,
        far_stats="mean,count,mean_r,rms_r",
        far_source_norm=False,
        far_gate_init=0.0,
        cutoff_radius=1.0,
    ).to(dtype=torch.float64)

    local_pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [2.9, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    local_source = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.5, 1.0, 0.0],
            [1.5, 0.5, 1.0],
            [0.0, 2.0, 0.5],
        ],
        dtype=torch.float64,
    )

    shell_bounds = module._shell_bounds(device=local_pos.device, dtype=local_pos.dtype)
    sparse_shells, sparse_sum, sparse_count = module._compute_graph_shell_features(local_pos, local_source)
    dense_shells, dense_sum, dense_count = _manual_shell_stats(
        module,
        local_pos,
        local_source,
        cutoff=1.0,
        shell_bounds=shell_bounds,
    )

    assert torch.allclose(sparse_shells, dense_shells, atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(sparse_sum, dense_sum, atol=1.0e-10, rtol=1.0e-10)
    assert torch.allclose(sparse_count, dense_count, atol=1.0e-10, rtol=1.0e-10)


def test_isolated_far_field_v2_kernel_starts_as_physical_base() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v2",
        feature_dim=6,
        hidden_dim=8,
        boundary="nonperiodic",
        neutralize=False,
        energy_partition="potential",
        far_source_dim=3,
        far_num_shells=2,
        far_shell_growth=2.0,
        far_tail=True,
        far_stats="mean,count,mean_r,rms_r",
        far_source_norm=False,
        far_gate_init=0.0,
        cutoff_radius=1.0,
    ).to(dtype=torch.float64)

    distance = torch.tensor([1.2, 2.0, 2.8], dtype=torch.float64)
    screening = torch.nn.functional.softplus(module.kernel_screening_raw.to(dtype=torch.float64))
    base_kernel = torch.exp(-screening * distance) / distance.clamp_min(1.0e-6)
    assert torch.allclose(module._physical_kernel(distance), base_kernel, atol=1.0e-12, rtol=1.0e-12)


def test_isolated_far_field_v2_tail_toggle_shape() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v2",
        feature_dim=10,
        hidden_dim=12,
        boundary="nonperiodic",
        neutralize=False,
        energy_partition="uniform",
        far_source_dim=4,
        far_num_shells=2,
        far_shell_growth=2.0,
        far_tail=False,
        far_stats="mean,count,mean_r,rms_r",
        far_source_norm=False,
        far_gate_init=0.0,
        cutoff_radius=1.0,
    ).to(dtype=torch.float64)

    x = torch.randn(4, 10, dtype=torch.float64)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [4.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    batch = torch.zeros(4, dtype=torch.long)
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    out = module(x, pos, batch, cell)
    assert out.shape == (4, 1)


def test_isolated_far_field_v2_derives_explicit_radius_from_shell_geometry() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v2",
        feature_dim=10,
        hidden_dim=12,
        boundary="nonperiodic",
        neutralize=False,
        energy_partition="uniform",
        far_source_dim=4,
        far_num_shells=3,
        far_shell_growth=2.0,
        far_tail=True,
        far_stats="mean,count,mean_r,rms_r",
        far_source_norm=False,
        far_gate_init=0.0,
        cutoff_radius=1.0,
    ).to(dtype=torch.float64)

    assert module.far_max_radius_multiplier == pytest.approx(8.0)


def test_isolated_far_field_v2_uses_configured_tail_bin_count() -> None:
    module = build_long_range_module(
        mode="isolated-far-field-v2",
        feature_dim=10,
        hidden_dim=12,
        boundary="nonperiodic",
        neutralize=False,
        energy_partition="uniform",
        far_source_dim=4,
        far_num_shells=2,
        far_shell_growth=2.0,
        far_tail=True,
        far_tail_bins=3,
        far_stats="mean,count,mean_r,rms_r",
        far_source_norm=False,
        far_gate_init=0.0,
        cutoff_radius=1.0,
    ).to(dtype=torch.float64)

    tail_bounds = module._tail_bin_bounds()
    assert module.tail_num_bins == 3
    assert module.total_shells == 5
    assert len(tail_bounds) == 3
    assert tail_bounds[0] == (4.0, 8.0)
    assert tail_bounds[1] == (8.0, 16.0)
    assert tail_bounds[2][0] == 16.0
    assert tail_bounds[2][1] == float("inf")


def test_isolated_far_field_v2_rejects_invalid_tail_bin_count() -> None:
    with pytest.raises(ValueError, match="far_tail_bins >= 1"):
        build_long_range_module(
            mode="isolated-far-field-v2",
            feature_dim=10,
            hidden_dim=12,
            boundary="nonperiodic",
            neutralize=False,
            energy_partition="uniform",
            far_source_dim=4,
            far_num_shells=2,
            far_shell_growth=2.0,
            far_tail=True,
            far_tail_bins=0,
            far_stats="mean,count,mean_r,rms_r",
            far_source_norm=False,
            far_gate_init=0.0,
            cutoff_radius=1.0,
        )


def test_isolated_far_field_v2_rejects_empty_shell_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="far_shell_growth \\*\\* far_num_shells <= far_max_radius_multiplier",
    ):
        build_long_range_module(
            mode="isolated-far-field-v2",
            feature_dim=10,
            hidden_dim=12,
            boundary="nonperiodic",
            neutralize=False,
            energy_partition="uniform",
            far_source_dim=4,
            far_num_shells=3,
            far_shell_growth=2.0,
            far_tail=False,
            far_stats="mean,count,mean_r,rms_r",
            far_max_radius_multiplier=6.0,
            far_source_norm=False,
            far_gate_init=0.0,
            cutoff_radius=1.0,
        )


def test_export_core_writes_isolated_far_field_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="mff-isolated-far-export-") as td:
        tmpdir = Path(td)
        ckpt = tmpdir / "isolated_far_field.pth"
        out_pt = tmpdir / "core.pt"
        _make_dummy_checkpoint_pure_cartesian_ictd(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="isolated-far-field-v1",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_source_channels=8,
            long_range_energy_partition="potential",
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
            export_reciprocal_source=False,
        )

        meta = json.loads(out_pt.with_suffix(".pt.json").read_text(encoding="utf-8"))
        assert meta["export_reciprocal_source"] is False
        assert meta["long_range_runtime_backend"] == "none"
        assert meta["long_range_boundary"] == "nonperiodic"
        assert meta["long_range_backend"] == "dense_pairwise"
        assert meta["long_range_energy_partition"] == "potential"
        assert meta["long_range_source_channels"] == 0


def test_export_core_writes_isolated_far_field_v2_metadata() -> None:
    with tempfile.TemporaryDirectory(prefix="mff-isolated-far-v2-export-") as td:
        tmpdir = Path(td)
        ckpt = tmpdir / "isolated_far_field_v2.pth"
        out_pt = tmpdir / "core.pt"
        _make_dummy_checkpoint_pure_cartesian_ictd(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="isolated-far-field-v2",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_energy_partition="potential",
            long_range_far_source_dim=6,
            long_range_far_num_shells=2,
            long_range_far_shell_growth=2.0,
            long_range_far_tail=True,
            long_range_far_tail_bins=3,
            long_range_far_stats="mean,count,mean_r,rms_r",
            long_range_far_source_norm=True,
            long_range_far_gate_init=0.0,
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
            export_reciprocal_source=False,
        )

        meta = json.loads(out_pt.with_suffix(".pt.json").read_text(encoding="utf-8"))
        assert meta["export_reciprocal_source"] is False
        assert meta["long_range_runtime_backend"] == "none"
        assert meta["long_range_boundary"] == "nonperiodic"
        assert meta["long_range_backend"] == "dense_pairwise"


def test_spherical_save_cue_dummy_checkpoint_supports_isolated_far_field() -> None:
    pytest.importorskip("cuequivariance_torch")
    with tempfile.TemporaryDirectory(prefix="mff-isolated-far-cue-") as td:
        ckpt = Path(td) / "isolated_far_field_cue.pth"
        _make_dummy_checkpoint_spherical_save_cue(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="isolated-far-field-v1",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_source_channels=8,
        )
        assert ckpt.exists()


def test_spherical_save_cue_dummy_checkpoint_supports_isolated_far_field_v2() -> None:
    pytest.importorskip("cuequivariance_torch")
    with tempfile.TemporaryDirectory(prefix="mff-isolated-far-v2-cue-") as td:
        ckpt = Path(td) / "isolated_far_field_v2_cue.pth"
        _make_dummy_checkpoint_spherical_save_cue(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="isolated-far-field-v2",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_energy_partition="potential",
            long_range_far_source_dim=6,
            long_range_far_num_shells=2,
            long_range_far_shell_growth=2.0,
            long_range_far_tail=True,
            long_range_far_tail_bins=3,
            long_range_far_stats="mean,count,mean_r,rms_r",
            long_range_far_source_norm=True,
            long_range_far_gate_init=0.0,
        )
        assert ckpt.exists()


def test_export_core_spherical_save_cue_supports_isolated_far_field() -> None:
    pytest.importorskip("cuequivariance_torch")
    with tempfile.TemporaryDirectory(prefix="mff-isolated-far-cue-export-") as td:
        tmpdir = Path(td)
        ckpt = tmpdir / "isolated_far_field_cue.pth"
        out_pt = tmpdir / "core.pt"
        _make_dummy_checkpoint_spherical_save_cue(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="isolated-far-field-v1",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_source_channels=8,
            long_range_energy_partition="potential",
        )

        export_core(
            checkpoint=str(ckpt),
            elements=["H", "O"],
            device="cpu",
            max_radius=None,
            num_interaction=None,
            out_pt=str(out_pt),
            tensor_product_mode="spherical-save-cue",
            embed_e0=False,
            export_reciprocal_source=False,
        )

        meta = json.loads(out_pt.with_suffix(".pt.json").read_text(encoding="utf-8"))
        assert meta["export_reciprocal_source"] is False
        assert meta["long_range_runtime_backend"] == "none"
        assert meta["long_range_boundary"] == "nonperiodic"


def test_export_core_spherical_save_cue_supports_isolated_far_field_v2() -> None:
    pytest.importorskip("cuequivariance_torch")
    with tempfile.TemporaryDirectory(prefix="mff-isolated-far-v2-cue-export-") as td:
        tmpdir = Path(td)
        ckpt = tmpdir / "isolated_far_field_v2_cue.pth"
        out_pt = tmpdir / "core.pt"
        _make_dummy_checkpoint_spherical_save_cue(
            str(ckpt),
            device=torch.device("cpu"),
            long_range_mode="isolated-far-field-v2",
            long_range_boundary="nonperiodic",
            long_range_hidden_dim=16,
            long_range_energy_partition="potential",
            long_range_far_source_dim=6,
            long_range_far_num_shells=2,
            long_range_far_shell_growth=2.0,
            long_range_far_tail=True,
            long_range_far_tail_bins=3,
            long_range_far_stats="mean,count,mean_r,rms_r",
            long_range_far_source_norm=True,
            long_range_far_gate_init=0.0,
        )

        export_core(
            checkpoint=str(ckpt),
            elements=["H", "O"],
            device="cpu",
            max_radius=None,
            num_interaction=None,
            out_pt=str(out_pt),
            tensor_product_mode="spherical-save-cue",
            embed_e0=False,
            export_reciprocal_source=False,
        )

        meta = json.loads(out_pt.with_suffix(".pt.json").read_text(encoding="utf-8"))
        assert meta["export_reciprocal_source"] is False
        assert meta["long_range_runtime_backend"] == "none"
        assert meta["long_range_boundary"] == "nonperiodic"
