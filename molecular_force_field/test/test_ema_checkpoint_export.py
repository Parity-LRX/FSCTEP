from __future__ import annotations

import copy

import torch

from molecular_force_field.cli.export_libtorch_core import export_core
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
)
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.utils.checkpoint_metadata import get_checkpoint_e3_state_dict


def _make_core_inputs(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    n_atoms, n_edges = 12, 48
    pos = torch.zeros(n_atoms, 3, device=device, dtype=dtype, requires_grad=True)
    atom_types = torch.ones(n_atoms, device=device, dtype=torch.long)
    batch = torch.zeros(n_atoms, device=device, dtype=torch.long)
    edge_src = torch.randint(0, n_atoms, (n_edges,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, n_atoms, (n_edges,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(n_edges, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0
    edge_vec = (pos[edge_dst] - pos[edge_src] + torch.randn(n_edges, 3, device=device, dtype=dtype)).detach()
    external_tensor = torch.zeros(3, device=device, dtype=dtype)
    return pos, atom_types, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor


def _run_core_forward(core: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    with torch.no_grad():
        out = core(*inputs)
    return out[0] if isinstance(out, tuple) else out


def _trace_checkpoint_core(checkpoint_path: str) -> torch.nn.Module:
    mliap = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=checkpoint_path,
        element_types=["H", "O"],
        max_radius=5.0,
        atomic_energy_keys=[1, 8],
        atomic_energy_values=[-13.6, -75.0],
        device="cpu",
        torchscript=True,
    )
    core = mliap.wrapper.model.core
    core.eval()
    return core


def test_get_checkpoint_e3_state_dict_prefers_ema():
    checkpoint = {
        "e3trans_state_dict": {"weight": torch.tensor([0.0])},
        "e3trans_ema_state_dict": {"weight": torch.tensor([1.0])},
    }
    state_dict, source = get_checkpoint_e3_state_dict(checkpoint)
    assert source == "ema"
    assert torch.equal(state_dict["weight"], torch.tensor([1.0]))


def test_trainer_auto_enables_ema_validation_and_checkpoint_saving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trainer = Trainer(
        model=torch.nn.Linear(1, 1),
        e3trans=torch.nn.Linear(1, 1),
        train_loader=[0],
        val_loader=[0],
        train_dataset=[0],
        val_dataset=[0],
        device=torch.device("cpu"),
        ema_start_epoch=1,
        checkpoint_path="dummy.pth",
    )
    assert trainer.use_ema_for_validation is True
    assert trainer.save_ema_model is True


def test_export_core_prefers_ema_checkpoint_weights(tmp_path):
    torch.manual_seed(0)
    mixed_ckpt = tmp_path / "mixed_ema_checkpoint.pth"
    raw_ckpt = tmp_path / "raw_checkpoint.pth"
    ema_ckpt = tmp_path / "ema_checkpoint.pth"
    out_pt = tmp_path / "core.pt"

    _make_dummy_checkpoint_pure_cartesian_ictd(
        str(mixed_ckpt),
        device=torch.device("cpu"),
        external_tensor_rank=1,
    )
    checkpoint = torch.load(mixed_ckpt, map_location="cpu", weights_only=False)

    zero_state = {
        key: torch.zeros_like(value) if torch.is_tensor(value) else value
        for key, value in checkpoint["e3trans_state_dict"].items()
    }
    ema_state = copy.deepcopy(zero_state)
    for key, value in ema_state.items():
        if torch.is_tensor(value) and value.ndim == 1:
            ema_state[key] = torch.full_like(value, 0.125)

    checkpoint["e3trans_state_dict"] = zero_state
    checkpoint["e3trans_ema_state_dict"] = ema_state
    checkpoint["ema_enabled"] = True
    torch.save(checkpoint, mixed_ckpt)

    raw_only_checkpoint = copy.deepcopy(checkpoint)
    raw_only_checkpoint.pop("e3trans_ema_state_dict", None)
    torch.save(raw_only_checkpoint, raw_ckpt)

    ema_only_checkpoint = copy.deepcopy(checkpoint)
    ema_only_checkpoint["e3trans_state_dict"] = ema_state
    ema_only_checkpoint.pop("e3trans_ema_state_dict", None)
    torch.save(ema_only_checkpoint, ema_ckpt)

    export_core(
        checkpoint=str(mixed_ckpt),
        elements=["H", "O"],
        device="cpu",
        max_radius=5.0,
        num_interaction=2,
        out_pt=str(out_pt),
        tensor_product_mode="pure-cartesian-ictd",
        embed_e0=False,
    )

    exported_core = torch.jit.load(str(out_pt), map_location="cpu")
    exported_core.eval()
    raw_core = _trace_checkpoint_core(str(raw_ckpt))
    ema_core = _trace_checkpoint_core(str(ema_ckpt))

    inputs = _make_core_inputs(torch.device("cpu"), torch.float32)
    exported_energy = _run_core_forward(exported_core, inputs)
    raw_energy = _run_core_forward(raw_core, inputs)
    ema_energy = _run_core_forward(ema_core, inputs)

    assert torch.allclose(exported_energy, ema_energy, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(exported_energy, raw_energy, atol=1e-6, rtol=1e-6)
