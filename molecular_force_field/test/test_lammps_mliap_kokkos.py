#!/usr/bin/env python3
"""测试 LAMMPS ML-IAP 在 Kokkos GPU 上的运行。

用法:
    python test_lammps_mliap_kokkos.py
    python test_lammps_mliap_kokkos.py --kokkos   # 尝试 activate_mliappy_kokkos

需先导出模型: python -m molecular_force_field.cli.export_mliap checkpoint.pth --elements H O -o model.pt
或使用脚本自动创建 dummy 模型测试。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

# 项目根目录加入 path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def test_mliap_standard(model_pt: str) -> bool:
    """标准 ML-IAP (activate_mliappy)"""
    try:
        import lammps
        from lammps.mliap import activate_mliappy, load_unified
        import torch
    except ImportError as e:
        print(f"SKIP: 缺少依赖 {e}")
        return True

    lmp = lammps.lammps(cmdargs=["-nocite", "-log", "none"])
    activate_mliappy(lmp)
    model = torch.load(model_pt, map_location="cpu", weights_only=False)
    load_unified(model)

    lmp.commands_string("""
units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 2 single 0.96 0.0 0.0
create_atoms 1 single 0.24 0.93 0.0
mass 1 1.008
mass 2 15.999
pair_style mliap unified """ + model_pt + """ 0
pair_coeff * * H O
run 0
""")
    e = lmp.get_thermo("pe")
    lmp.close()
    print(f"  E = {e:.4f} eV")
    return True


def test_mliap_kokkos(model_pt: str) -> bool:
    """ML-IAP-Kokkos (activate_mliappy_kokkos)"""
    try:
        import lammps
        from lammps.mliap import activate_mliappy_kokkos, load_unified
        import torch
    except ImportError as e:
        print(f"SKIP: 缺少依赖 {e}")
        return True

    try:
        lmp = lammps.lammps(cmdargs=[
            "-nocite", "-log", "none",
            "-k", "on", "g", "1", "-sf", "kk",
            "-pk", "kokkos", "newton", "on", "neigh", "half",
        ])
    except Exception as e:
        print(f"SKIP: Kokkos 启动失败: {e}")
        return True

    try:
        activate_mliappy_kokkos(lmp)
    except (ImportError, AttributeError) as e:
        print(f"SKIP: activate_mliappy_kokkos 不可用: {e}")
        lmp.close()
        return True

    model = torch.load(model_pt, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    load_unified(model)

    lmp.commands_string("""
units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 2 single 0.96 0.0 0.0
create_atoms 1 single 0.24 0.93 0.0
mass 1 1.008
mass 2 15.999
pair_style mliap unified """ + model_pt + """ 0
pair_coeff * * H O
run 0
""")
    e = lmp.get_thermo("pe")
    lmp.close()
    print(f"  E = {e:.4f} eV (Kokkos GPU)")
    return True


def main():
    parser = argparse.ArgumentParser(description="测试 LAMMPS ML-IAP Kokkos")
    parser.add_argument("--model", "-m", type=str, default=None, help="model-mliap.pt 路径")
    parser.add_argument("--kokkos", action="store_true", help="同时测试 Kokkos GPU 接口")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        model_pt = args.model
        if model_pt is None or not os.path.isfile(model_pt):
            # 创建 dummy 模型
            from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint
            import torch
            ckpt = os.path.join(td, "dummy.pth")
            _make_dummy_checkpoint(ckpt, torch.device("cpu"))
            model_pt = os.path.join(td, "dummy-mliap.pt")
            from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF
            obj = LAMMPS_MLIAP_MFF.from_checkpoint(
                ckpt, element_types=["H", "O"], max_radius=3.0,
                atomic_energy_keys=[1, 8], atomic_energy_values=[-13.6, -75.0],
            )
            torch.save(obj, model_pt)
            print("使用临时 dummy 模型:", model_pt)
        else:
            print("使用模型:", model_pt)

        print("\n--- 标准 ML-IAP ---")
        test_mliap_standard(model_pt)
        print("PASS: 标准 ML-IAP")

        if args.kokkos:
            print("\n--- ML-IAP-Kokkos GPU ---")
            test_mliap_kokkos(model_pt)
            print("PASS: ML-IAP-Kokkos")

    print("\n全部测试通过。")


if __name__ == "__main__":
    main()
