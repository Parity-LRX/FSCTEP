#!/usr/bin/env python3
"""GPU 自测：PureCartesianICTD (pure-cartesian-ictd) TorchScript + ML-IAP-Kokkos.

这个脚本在 GPU 机器上做 2 类验证：
1) TorchScript(trace) 导出 + 反序列化 + 多组 (N,E) 的前向/力计算（不依赖 LAMMPS）
2) （可选）通过 Python 驱动 LAMMPS + Kokkos + ML-IAP unified 跑最小 run，确认回调不报错

用法示例：
  # 只做 TorchScript 自测（推荐先跑这个）
  python molecular_force_field/scripts/test_torchscript_mliap_kokkos.py

  # 用真实 checkpoint（建议你的实际任务也跑一遍）
  python molecular_force_field/scripts/test_torchscript_mliap_kokkos.py --checkpoint ckpt.pth --elements H O

  # 额外跑 LAMMPS Kokkos 联机验证（需要已安装带 ML-IAP+PYTHON+Kokkos 的 LAMMPS Python 模块）
  python molecular_force_field/scripts/test_torchscript_mliap_kokkos.py --with-lammps
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from typing import List

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _pick_device(req: str) -> str:
    if req == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        return "cpu"
    return req


def _make_ckpt_if_needed(path: str | None, device: torch.device) -> str:
    if path is not None and os.path.isfile(path):
        return path
    from molecular_force_field.test.self_test_lammps_potential import (
        _make_dummy_checkpoint_pure_cartesian_ictd,
    )
    ckpt = os.path.join(tempfile.mkdtemp(prefix="mliap-ts-"), "dummy.pth")
    _make_dummy_checkpoint_pure_cartesian_ictd(ckpt, device=torch.device("cpu"))
    print(f"使用临时 dummy checkpoint: {ckpt}")
    return ckpt


def _export_torchscript_mliap(
    ckpt: str,
    elements: List[str],
    device: str,
    max_radius: float,
    num_interaction: int,
) -> str:
    from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

    obj = LAMMPS_MLIAP_MFF.from_checkpoint(
        checkpoint_path=ckpt,
        element_types=elements,
        max_radius=max_radius,
        atomic_energy_keys=[1, 8],
        atomic_energy_values=[-13.6, -75.0],
        device=device,
        tensor_product_mode="pure-cartesian-ictd",
        num_interaction=num_interaction,
        torchscript=True,
    )
    out = os.path.join(tempfile.mkdtemp(prefix="mliap-ts-"), "model-mliap-ts.pt")
    torch.save(obj, out)
    print(f"已导出 TorchScript ML-IAP 对象: {out}")
    return out


def _torchscript_smoketest(model_pt: str, device: str) -> None:
    print("\n--- 自测 1：TorchScript 前向/力（多 shape） ---")
    obj = torch.load(model_pt, map_location=device, weights_only=False)
    # wrapper/model 已在运行时根据 data device 再 .to(...)，这里先确保在目标 device
    obj.wrapper = obj.wrapper.to(device)
    obj._elem_to_Z = obj._elem_to_Z.to(device)

    # 多组 shape：模拟 MD 中 E 的变化
    cases = [(32, 128), (64, 512), (200, 4096), (512, 16384)]
    for (N, E) in cases:
        rij = torch.randn(E, 3, device=device, dtype=obj.dtype)
        A = torch.randint(0, 2, (N,), device=device, dtype=torch.long)
        species = obj._elem_to_Z[A]
        batch = torch.zeros(N, device=device, dtype=torch.long)
        edge_src = torch.randint(0, N, (E,), device=device, dtype=torch.long)
        edge_dst = torch.randint(0, N, (E,), device=device, dtype=torch.long)
        edge_shifts = torch.zeros(E, 3, device=device, dtype=obj.dtype)
        cell = torch.eye(3, device=device, dtype=obj.dtype).unsqueeze(0) * 100.0

        t0 = time.perf_counter()
        E_total, atom_e, atom_f = obj.wrapper(
            rij, species, batch, edge_src, edge_dst, edge_shifts, cell, nlocal=N
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        assert torch.isfinite(E_total).item()
        assert atom_e.shape[0] == N
        assert atom_f.shape == (N, 3)
        assert torch.isfinite(atom_f).all().item()
        print(f"  PASS (N={N}, E={E})  {((t1 - t0) * 1000):.2f} ms")

    print("PASS: TorchScript 前向/力自测通过")


def _lammps_kokkos_smoketest(model_pt: str, skin: float, n_atoms: int, n_steps: int) -> None:
    print("\n--- 自测 2：LAMMPS + Kokkos + ML-IAP 联机 ---")
    try:
        import lammps  # type: ignore
        from lammps.mliap import activate_mliappy, load_unified  # type: ignore
        from lammps.mliap import activate_mliappy_kokkos  # type: ignore
    except Exception as e:
        print(f"SKIP: LAMMPS Python/Kokkos 不可用: {e}")
        return

    lmp = None
    try:
        lmp = lammps.lammps(cmdargs=[
            "-nocite", "-log", "none",
            "-k", "on", "g", "1", "-sf", "kk",
            "-pk", "kokkos", "newton", "on", "neigh", "half",
        ])
        activate_mliappy(lmp)
        activate_mliappy_kokkos(lmp)

        model = torch.load(model_pt, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        load_unified(model)

        # 随机 H/O
        box = max(25.0, 2.5 * (n_atoms ** (1 / 3)))
        n1 = (2 * n_atoms) // 3  # H
        n2 = n_atoms - n1        # O
        if n2 <= 0:
            n2 = 1
            n1 = n_atoms - 1

        cmds = f"""
units metal
atom_style atomic
boundary p p p
region box block 0 {box} 0 {box} 0 {box}
create_box 2 box
create_atoms 1 random {n1} 12345 box
create_atoms 2 random {n2} 12346 box
mass 1 1.008
mass 2 15.999
neighbor {skin} bin
pair_style mliap unified {model_pt} 0
pair_coeff * * H O
velocity all create 300 42
fix 1 all nve
"""
        lmp.commands_string(cmds.strip())

        lmp.command("run 2")
        lmp.command(f"run {n_steps}")
        e = lmp.get_thermo("pe")
        print(f"  PASS: LAMMPS run OK, pe={e:.6f} eV")
    finally:
        try:
            if lmp is not None:
                lmp.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="GPU 自测：TorchScript + ML-IAP-Kokkos")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型 checkpoint (.pth)；不提供则生成 dummy")
    parser.add_argument("--elements", nargs="+", default=["H", "O"], help="元素顺序（LAMMPS type 顺序）")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max-radius", type=float, default=3.0, help="cutoff (Angstrom)")
    parser.add_argument("--num-interaction", type=int, default=2)
    parser.add_argument("--with-lammps", action="store_true", help="额外运行 LAMMPS+Kokkos 联机验证")
    parser.add_argument("--skin", type=float, default=1.0, help="LAMMPS neighbor skin (Angstrom)")
    parser.add_argument("--n-atoms", type=int, default=2000, help="联机测试原子数")
    parser.add_argument("--n-steps", type=int, default=4, help="联机测试步数（不含预热 run 2）")
    args = parser.parse_args()

    device = _pick_device(args.device)
    ckpt = _make_ckpt_if_needed(args.checkpoint, device=torch.device(device))
    model_pt = _export_torchscript_mliap(
        ckpt=ckpt,
        elements=args.elements,
        device=device,
        max_radius=float(args.max_radius),
        num_interaction=int(args.num_interaction),
    )
    _torchscript_smoketest(model_pt, device=device)
    if args.with_lammps:
        _lammps_kokkos_smoketest(model_pt, skin=float(args.skin), n_atoms=int(args.n_atoms), n_steps=int(args.n_steps))

    print("\n全部自测完成。")


if __name__ == "__main__":
    main()

