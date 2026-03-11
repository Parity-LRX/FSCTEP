from __future__ import annotations

import torch

from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint
from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
)


def test_build_e3trans_from_checkpoint_uses_checkpoint_atomic_energies(tmp_path) -> None:
    ckpt_path = tmp_path / "dummy_ictd.pth"
    _make_dummy_checkpoint_pure_cartesian_ictd(
        str(ckpt_path),
        device=torch.device("cpu"),
    )
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    ckpt["atomic_energy_keys"] = torch.tensor([1, 8], dtype=torch.long)
    ckpt["atomic_energy_values"] = torch.tensor([-13.6, -75.0], dtype=torch.float64)
    torch.save(ckpt, str(ckpt_path))

    _, config = build_e3trans_from_checkpoint(
        str(ckpt_path),
        torch.device("cpu"),
    )

    assert config.atomic_energy_keys.tolist() == [1, 8]
    assert config.atomic_energy_values.tolist() == [-13.6, -75.0]


def test_build_e3trans_from_checkpoint_atomic_energy_file_overrides_checkpoint(tmp_path) -> None:
    ckpt_path = tmp_path / "dummy_ictd.pth"
    e0_csv = tmp_path / "fitted_E0.csv"
    _make_dummy_checkpoint_pure_cartesian_ictd(
        str(ckpt_path),
        device=torch.device("cpu"),
    )
    e0_csv.write_text("Atom,E0\n1,-0.1\n8,-0.8\n", encoding="utf-8")

    _, config = build_e3trans_from_checkpoint(
        str(ckpt_path),
        torch.device("cpu"),
        atomic_energy_file=str(e0_csv),
    )

    assert config.atomic_energy_keys.tolist() == [1, 8]
    assert config.atomic_energy_values.tolist() == [-0.1, -0.8]


def test_build_e3trans_from_checkpoint_keeps_legacy_default_fallback(tmp_path) -> None:
    ckpt_path = tmp_path / "legacy_dummy_ictd.pth"
    _make_dummy_checkpoint_pure_cartesian_ictd(
        str(ckpt_path),
        device=torch.device("cpu"),
    )

    _, config = build_e3trans_from_checkpoint(
        str(ckpt_path),
        torch.device("cpu"),
    )

    assert config.atomic_energy_keys.tolist() == [1, 6, 7, 8]
    assert config.atomic_energy_values.tolist() == [-430.53, -821.03, -1488.19, -2044.35]
