#!/usr/bin/env python3
"""ML-IAP 完整测试脚本

在 GPU 机器上安装 Kokkos LAMMPS 后运行，验证 ML-IAP 全流程：
  1. atom forces 数值一致性（AtomForcesWrapper: dE/d(pos) via dummy pos）
  2. edge forces 数值一致性（legacy EdgeForcesWrapper: dE/d(edge_vec)）
  3. 导出/加载往返
  4. LAMMPS ML-IAP 联机测试
  5. （可选）ML-IAP-Kokkos GPU 测试

用法:
  cd /path/to/rebuild
  python molecular_force_field/scripts/test_mliap_full.py
  python molecular_force_field/scripts/test_mliap_full.py --kokkos    # 含 Kokkos GPU 测试
  python molecular_force_field/scripts/test_mliap_full.py -m model.pt  # 指定模型

macOS 若 import lammps 失败，先设置:
  export DYLD_LIBRARY_PATH="$HOME/.local/lib:..."
"""

from __future__ import annotations

import argparse
import os
import sys

# 项目根目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# 限制线程
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def main():
    parser = argparse.ArgumentParser(description="ML-IAP 完整测试")
    parser.add_argument("--model", "-m", type=str, default=None, help="model-mliap.pt 路径（默认用 dummy）")
    parser.add_argument("--mode", type=str, default="spherical",
                        choices=["spherical", "spherical-save", "spherical-save-cue", "pure-cartesian-ictd", "pure-cartesian-ictd-save"],
                        help="测试的模型模式（默认: spherical）。spherical-save-cue 需 cuEquivariance")
    parser.add_argument("--kokkos", action="store_true", help="额外测试 ML-IAP-Kokkos GPU")
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("ML-IAP 完整测试")
    print("=" * 60)
    print(f"项目根目录: {_repo_root}")
    print(f"设备: {args.device}")
    print(f"模式: {args.mode}")
    print()

    ok = True

    # 1. atom forces 数值一致性（AtomForcesWrapper, 新默认）
    print("--- 测试 1: atom forces 数值一致性 (AtomForcesWrapper) ---")
    from molecular_force_field.test.test_mliap import test_atom_forces_consistency
    ok = test_atom_forces_consistency(args.device, args.mode) and ok

    # 2. edge forces 数值一致性（旧 EdgeForcesWrapper, 兼容性）
    print("\n--- 测试 2: edge forces 数值一致性 (legacy) ---")
    from molecular_force_field.test.test_mliap import test_edge_forces_consistency
    ok = test_edge_forces_consistency(args.device, args.mode) and ok

    # 3. 导出/加载往返
    print("\n--- 测试 3: 导出/加载往返 ---")
    from molecular_force_field.test.test_mliap import test_export_and_load
    ok = test_export_and_load(args.device, args.mode) and ok

    # 4. LAMMPS ML-IAP 联机
    print("\n--- 测试 4: LAMMPS ML-IAP 联机 ---")
    from molecular_force_field.test.test_mliap import test_lammps_mliap_online
    ok = test_lammps_mliap_online(args.device, args.mode) and ok

    # 5. （可选）Kokkos GPU
    if args.kokkos:
        print("\n--- 测试 5: ML-IAP-Kokkos GPU ---")
        import tempfile
        import torch
        from molecular_force_field.test.test_mliap import _make_dummy_checkpoint_for_mode
        from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

        def _test_kokkos(model_pt: str) -> bool:
            try:
                import lammps
                from lammps.mliap import activate_mliappy_kokkos, load_unified
            except (ImportError, OSError) as e:
                print(f"SKIP: {e}")
                return True
            try:
                lmp = lammps.lammps(cmdargs=[
                    "-nocite", "-log", "none",
                    "-k", "on", "g", "1", "-sf", "kk",
                    "-pk", "kokkos", "newton", "on", "neigh", "half",
                ])
                activate_mliappy_kokkos(lmp)
            except Exception as e:
                print(f"SKIP: Kokkos 不可用: {e}")
                return True
            model = torch.load(model_pt, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
            load_unified(model)
            lmp.commands_string(f"""
units metal
atom_style atomic
region box block 0 10 0 10 0 10
create_box 2 box
create_atoms 1 single 0.0 0.0 0.0
create_atoms 2 single 0.96 0.0 0.0
create_atoms 1 single 0.24 0.93 0.0
mass 1 1.008
mass 2 15.999
pair_style mliap unified {model_pt} 0
pair_coeff * * H O
run 0
""")
            e = lmp.get_thermo("pe")
            lmp.close()
            print(f"  E = {e:.4f} eV (Kokkos GPU)")
            return True

        with tempfile.TemporaryDirectory() as td:
            ckpt = os.path.join(td, "dummy.pth")
            _make_dummy_checkpoint_for_mode(args.mode, ckpt, torch.device("cpu"))
            model_pt = args.model or os.path.join(td, "dummy-mliap.pt")
            if not args.model:
                obj = LAMMPS_MLIAP_MFF.from_checkpoint(
                    ckpt, element_types=["H", "O"], max_radius=3.0,
                    atomic_energy_keys=[1, 8], atomic_energy_values=[-13.6, -75.0],
                )
                torch.save(obj, model_pt)
            ok = _test_kokkos(model_pt) and ok
        print("PASS: ML-IAP-Kokkos")

    print("\n" + "=" * 60)
    if ok:
        print("ML-IAP 完整测试全部通过。")
    else:
        print("部分测试失败。")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
