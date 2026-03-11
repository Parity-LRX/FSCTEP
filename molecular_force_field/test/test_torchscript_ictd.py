#!/usr/bin/env python3
"""
测试 pure-cartesian-ictd 模型能否成功 TorchScript trace 并运行。

包含：
1. 直接 trace 测试（LAMMPS_MLIAP_MFF.from_checkpoint + torchscript=True）
2. 完整导出测试（export_core -> core.pt -> torch.jit.load 加载并 forward）

Usage:
  python -m molecular_force_field.test.test_torchscript_ictd
"""

from __future__ import annotations

import tempfile

import torch

from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint_pure_cartesian_ictd,
)
from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF


def _run_forward(
    core: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    *,
    external_tensor_rank: int | None = None,
):
    """运行一次 forward，返回输出。"""
    N, E = 32, 256
    pos = torch.zeros(N, 3, device=device, dtype=dtype, requires_grad=True)
    A = torch.ones(N, device=device, dtype=torch.long)
    batch = torch.zeros(N, device=device, dtype=torch.long)
    edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
    edge_shifts = torch.zeros(E, 3, device=device, dtype=dtype)
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0) * 100.0
    edge_vec = (pos[edge_dst] - pos[edge_src] + torch.randn(E, 3, device=device, dtype=dtype)).detach()
    if external_tensor_rank is None:
        external_tensor = torch.empty(0, device=device, dtype=dtype)
    else:
        external_tensor = torch.zeros((3,) * int(external_tensor_rank), device=device, dtype=dtype)

    with torch.no_grad():
        out = core(pos, A, batch, edge_src, edge_dst, edge_shifts, cell, edge_vec, external_tensor)
    return out


def _atom_energy_from_output(out) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def test_trace_only(external_tensor_rank: int = 1):
    """测试：直接 trace 并运行 forward。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = f"{tmp}/dummy_ictd.pth"
        _make_dummy_checkpoint_pure_cartesian_ictd(
            ckpt_path, device=torch.device("cpu"), external_tensor_rank=external_tensor_rank
        )

        print("[test_trace_only] Loading model with TorchScript trace...")
        mliap = LAMMPS_MLIAP_MFF.from_checkpoint(
            checkpoint_path=ckpt_path,
            element_types=["H", "O"],
            max_radius=5.0,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            device=str(device),
            torchscript=True,
        )

        adapter = mliap.wrapper.model
        core = adapter.core
        assert isinstance(core, torch.jit.ScriptModule), "Expected ScriptModule"

        print(f"[test_trace_only] TorchScript trace OK for rank-{external_tensor_rank}. Running forward...")
        out = _run_forward(core, device, dtype, external_tensor_rank=external_tensor_rank)
        atom_energy = _atom_energy_from_output(out)
        print(f"[test_trace_only] Forward OK. Atom energy shape: {atom_energy.shape}")
        print(f"PASS: TorchScript ICTD trace works for rank-{external_tensor_rank}")


def test_export_core(external_tensor_rank: int = 1):
    """测试：完整 export_core 导出 -> 加载 core.pt -> forward。"""
    import os
    from molecular_force_field.cli.export_libtorch_core import export_core

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = f"{tmp}/dummy_ictd.pth"
        out_pt = f"{tmp}/core.pt"
        e0_csv = os.path.join(tmp, "e0.csv")

        _make_dummy_checkpoint_pure_cartesian_ictd(
            ckpt_path, device=torch.device("cpu"), external_tensor_rank=external_tensor_rank
        )

        # 创建最小 E0 CSV，供 export_core 使用
        with open(e0_csv, "w") as f:
            f.write("Atom,E0\n1,-13.6\n8,-75.0\n")

        print(f"[test_export_core] Running export_core for rank-{external_tensor_rank}...")
        export_core(
            checkpoint=ckpt_path,
            elements=["H", "O"],
            device=str(device),
            max_radius=5.0,
            num_interaction=2,
            out_pt=out_pt,
            tensor_product_mode="pure-cartesian-ictd",
            embed_e0=True,
            e0_csv=e0_csv,
        )

        print("[test_export_core] Loading core.pt...")
        core = torch.jit.load(out_pt, map_location=device)
        core.eval()

        print("[test_export_core] Running forward on loaded core...")
        out = _run_forward(core, device, dtype, external_tensor_rank=external_tensor_rank)
        atom_energy = _atom_energy_from_output(out)
        print(f"[test_export_core] Forward OK. Atom energy shape: {atom_energy.shape}")
        print(f"PASS: export_core -> core.pt -> load works for rank-{external_tensor_rank}")


def test_export_core_no_external_tensor():
    """测试：不带 external tensor 的 core.pt 导出/加载/forward。"""
    import json
    import os
    from molecular_force_field.cli.export_libtorch_core import export_core

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = f"{tmp}/dummy_ictd_no_field.pth"
        out_pt = f"{tmp}/core_no_field.pt"
        e0_csv = os.path.join(tmp, "e0.csv")

        _make_dummy_checkpoint_pure_cartesian_ictd(
            ckpt_path,
            device=torch.device("cpu"),
            external_tensor_rank=None,
        )

        with open(e0_csv, "w") as f:
            f.write("Atom,E0\n1,-13.6\n8,-75.0\n")

        export_core(
            checkpoint=ckpt_path,
            elements=["H", "O"],
            device=str(device),
            max_radius=5.0,
            num_interaction=2,
            out_pt=out_pt,
            tensor_product_mode="pure-cartesian-ictd",
            embed_e0=True,
            e0_csv=e0_csv,
        )

        with open(out_pt + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["external_tensor_rank"] is None

        core = torch.jit.load(out_pt, map_location=device)
        core.eval()
        out = _run_forward(core, device, dtype, external_tensor_rank=None)
        atom_energy = _atom_energy_from_output(out)
        assert atom_energy.shape == (32, 1)
        print("PASS: export_core works without external_tensor")


def test_export_core_physical_tensors(external_tensor_rank: int = 1):
    """测试：带 physical tensor heads 的 core.pt 导出后返回固定 schema tuple。"""
    import os
    from molecular_force_field.cli.export_libtorch_core import export_core

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    physical_tensor_outputs = {
        "dipole": {"ls": [1], "channels_out": 1, "reduce": "sum"},
        "dipole_per_atom": {"ls": [1], "channels_out": 1, "reduce": "none"},
        "polarizability": {"ls": [0, 2], "channels_out": 1, "reduce": "sum"},
        "polarizability_per_atom": {"ls": [0, 2], "channels_out": 1, "reduce": "none"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = f"{tmp}/dummy_ictd_phys.pth"
        out_pt = f"{tmp}/core_phys.pt"
        e0_csv = os.path.join(tmp, "e0.csv")

        _make_dummy_checkpoint_pure_cartesian_ictd(
            ckpt_path,
            device=torch.device("cpu"),
            external_tensor_rank=external_tensor_rank,
            physical_tensor_outputs=physical_tensor_outputs,
        )

        with open(e0_csv, "w") as f:
            f.write("Atom,E0\n1,-13.6\n8,-75.0\n")

        print(f"[test_export_core_physical_tensors] Running export_core for rank-{external_tensor_rank}...")
        export_core(
            checkpoint=ckpt_path,
            elements=["H", "O"],
            device=str(device),
            max_radius=5.0,
            num_interaction=2,
            out_pt=out_pt,
            tensor_product_mode="pure-cartesian-ictd",
            embed_e0=True,
            e0_csv=e0_csv,
        )

        core = torch.jit.load(out_pt, map_location=device)
        core.eval()
        out = _run_forward(core, device, dtype, external_tensor_rank=external_tensor_rank)
        assert isinstance(out, tuple), "Expected fixed-schema tuple output for physical tensor heads"
        assert len(out) == 5, "Expected (atom_energy, global_phys, atom_phys, global_mask, atom_mask)"

        atom_energy, global_phys, atom_phys, global_mask, atom_mask = out
        assert atom_energy.shape == (32, 1)
        assert global_phys.shape == (1, 22)
        assert atom_phys.shape == (32, 22)
        assert global_mask.shape == (4,)
        assert atom_mask.shape == (4,)
        assert global_mask.tolist() == [0.0, 1.0, 1.0, 0.0]
        assert atom_mask.tolist() == [0.0, 1.0, 1.0, 0.0]
        print(
            f"PASS: export_core physical tensor tuple schema works for rank-{external_tensor_rank}"
        )


def main():
    test_export_core_no_external_tensor()
    print()
    for rank in (1, 2):
        test_trace_only(external_tensor_rank=rank)
        print()
        test_export_core(external_tensor_rank=rank)
        print()
        test_export_core_physical_tensors(external_tensor_rank=rank)
        print()
    print("All TorchScript tests passed.")


if __name__ == "__main__":
    main()
