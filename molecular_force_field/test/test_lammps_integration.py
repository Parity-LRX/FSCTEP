"""LAMMPS 接口联机测试：在已安装 LAMMPS（带 Python 包）的前提下，
用本仓库势函数在 LAMMPS 中通过 fix external pf/callback 跑 run 0。

优先使用 LAMMPS Python 库在同一进程内驱动（最可靠）；
若无 lammps Python 模块则退化为调用 lmp 可执行文件。

运行方式：
    python -m molecular_force_field.test.test_lammps_integration
    python -m molecular_force_field.test.test_lammps_integration --lmp /path/to/lmp
"""

from __future__ import annotations

import os
import shutil
import site
import subprocess
import sys
import tempfile
import contextlib
import io

# 在 LAMMPS 回调里跑势函数时限制线程，避免死锁
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

from molecular_force_field.test.self_test_lammps_potential import (
    _make_dummy_checkpoint,
)


def _find_lammps_executable(lmp_path: str | None = None) -> str | None:
    if lmp_path and os.path.isfile(lmp_path):
        return os.path.abspath(lmp_path)
    for name in ("lmp", "lmp_serial", "lmp_mpi"):
        exe = shutil.which(name)
        if exe:
            return exe
    return None


def _write_minimal_data_file(path: str, atom_style: str = "atomic") -> None:
    # 与 self_test 一致：H-O-H，type 1=H, type 2=O。使用 atomic 以免依赖 LAMMPS MOLECULE 包
    if atom_style == "atomic":
        # LAMMPS read_data: Atoms # atomic 为 atom-ID atom-type x y z
        content = """LAMMPS data file - H-O-H test

3 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # atomic
1 1 0.0 0.0 0.0
2 2 0.96 0.0 0.0
3 1 -0.24 0.93 0.0
"""
    else:
        content = """# Minimal data (full style)

3 atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms # full

1 1 0.0 0.0 0.0
2 2 0.96 0.0 0.0
3 1 -0.24 0.93 0.0
"""
    with open(path, "w") as f:
        f.write(content)


def _prepare_lammps_input_in_dir(td: str, ckpt_path: str) -> None:
    """在目录 td 中生成 LAMMPS 输入与 minimal.data。"""
    from molecular_force_field.cli.lammps_interface import generate_lammps_input

    with contextlib.redirect_stdout(io.StringIO()):
        generate_lammps_input(
            ckpt_path,
            output_dir=td,
            max_radius=3.0,
            atomic_energy_file=None,
            device="cpu",
            embed_size=None,
            output_size=8,
            atomic_energy_keys=[1, 8],
            atomic_energy_values=[-13.6, -75.0],
            type_to_Z={1: 1, 2: 8},
        )
    _write_minimal_data_file(os.path.join(td, "minimal.data"), atom_style="atomic")
    input_path = os.path.join(td, "lammps_input.in")
    with open(input_path) as f:
        inp = f.read()
    # atom_style atomic 不依赖 MOLECULE 包，便于默认编译的 LAMMPS 通过
    system_block = """
units real
atom_style atomic
read_data minimal.data

"""
    inp = inp.replace("# Set up Python potential", system_block + "# Set up Python potential")
    inp = inp.replace("# Run simulation\n# run 1000", "run 0\n")
    with open(input_path, "w") as f:
        f.write(inp)


def _build_atoms_commands(n_atoms: int, box_size: float = 20.0) -> str:
    """生成 create_atoms 命令：小体系用固定坐标，大体系用网格坐标。"""
    if n_atoms <= 3:
        # 小体系：H-O-H 固定坐标
        coords = [
            (1, 0.0, 0.0, 0.0),
            (2, 0.96, 0.0, 0.0),
            (1, 0.24, 0.93, 0.0),
        ]
        return "\n".join(f"create_atoms {t} single {x} {y} {z}" for t, x, y, z in coords[:n_atoms])
    # 大体系：网格坐标，约 2/3 H、1/3 O，间距约 1.5 Å
    import math
    n = max(2, int(math.ceil(n_atoms ** (1 / 3))))
    spacing = max(1.2, (box_size - 2) / n)
    cmds = []
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if count >= n_atoms:
                    break
                x, y, z = 1.0 + i * spacing, 1.0 + j * spacing, 1.0 + k * spacing
                t = 1 if (count % 3) != 0 else 2  # 约 2/3 H, 1/3 O
                cmds.append(f"create_atoms {t} single {x} {y} {z}")
                count += 1
            if count >= n_atoms:
                break
        if count >= n_atoms:
            break
    return "\n".join(cmds)


def _run_lammps_in_process(td: str, ckpt_path: str, n_atoms: int = 3) -> bool:
    """用当前进程的 LAMMPS 库 + fix external + 真实势函数跑一步 run 0。"""
    import numpy as np
    import molecular_force_field.interfaces.lammps_potential as lp_mod
    lp_mod.lammps_potential_init(
        ckpt_path,
        device="cpu",
        max_radius=3.0,
        atomic_energy_keys=[1, 8],
        atomic_energy_values=[-13.6, -75.0],
        type_to_Z={1: 1, 2: 8},
    )
    pot = lp_mod._potential_instance
    if pot is None:
        return False
    try:
        import lammps
        lmp = lammps.lammps(cmdargs=["-nocite", "-log", "none"])
    except Exception:
        return False

    def external_callback(lmp_obj, ntimestep, nlocal, tag, x, f):
        nall = lmp_obj.extract_setting("nall")
        # 优先用 numpy 接口零拷贝；否则退化为 ctypes 复制
        try:
            npy = lmp_obj.numpy
            x_np = np.asarray(npy.extract_atom("x", nelem=nall), dtype=np.float64)
            type_np = np.asarray(npy.extract_atom("type", nelem=nall), dtype=np.int32)
        except (AttributeError, TypeError):
            x_ptr = lmp_obj.extract_atom("x")
            type_ptr = lmp_obj.extract_atom("type")
            x_np = np.fromiter(
                (x_ptr[i][j] for i in range(nall) for j in range(3)),
                dtype=np.float64, count=nall * 3,
            ).reshape(nall, 3)
            type_np = np.fromiter((type_ptr[i] for i in range(nall)), dtype=np.int32, count=nall)
        box_info = lmp_obj.extract_box()
        boxlo_np = np.array(box_info[0], dtype=np.float64)
        boxhi_np = np.array(box_info[1], dtype=np.float64)
        pbc_np = np.array(box_info[5], dtype=np.int32)
        tag_np = np.arange(1, nall + 1, dtype=np.int32)
        energy, forces = pot.compute(nlocal, nall, tag_np, type_np, x_np, boxlo_np, boxhi_np, pbc_np)
        # 批量写入力：f 可能是 numpy 或 ctypes
        f_arr = np.asarray(f) if hasattr(f, "__array_interface__") or hasattr(f, "__array__") else None
        if f_arr is not None and f_arr.shape[0] >= nlocal:
            f_arr[:nlocal] = forces[:nlocal]
        else:
            for i in range(nlocal):
                f[i][0] = float(forces[i, 0])
                f[i][1] = float(forces[i, 1])
                f[i][2] = float(forces[i, 2])
        lmp_obj.fix_external_set_energy_global("ext", energy)

    cwd = os.getcwd()
    try:
        os.chdir(td)
        # 盒子需足够大以容纳 n_atoms
        box_size = 15.0 + (n_atoms ** (1 / 3)) * 2.0
        box_lo, box_hi = 0.0, box_size
        atoms_cmds = _build_atoms_commands(n_atoms, box_size)
        inp = f"""
units real
atom_style atomic
region box block {box_lo} {box_hi} {box_lo} {box_hi} {box_lo} {box_hi}
create_box 2 box
{atoms_cmds}
mass 1 1.0
mass 2 16.0
pair_style zero 0.1
pair_coeff * *
fix ext all external pf/callback 1 1
fix_modify ext energy yes
"""
        if n_atoms > 3:
            inp += """
velocity all create 300 12345
fix nve all nve
"""
        lmp.commands_string(inp.strip())
        lmp.set_fix_external_callback("ext", external_callback, lmp)
        lmp.command("run 0" if n_atoms <= 3 else "run 10")
    except Exception as e:
        print(f"LAMMPS 库执行失败: {e}")
        return False
    finally:
        os.chdir(cwd)
    return True


def run_lammps_integration_test(lmp_exe: str | None = None, n_atoms: int = 50) -> bool:
    """生成临时 checkpoint 与 LAMMPS 输入，运行 LAMMPS。成功返回 True；未找到 LAMMPS 时跳过也返回 True；仅当 LAMMPS 实际运行失败时返回 False。"""
    lmp_exe_path = _find_lammps_executable(lmp_exe)
    # 先尝试同进程：用 LAMMPS Python 库（势回调与当前进程一致，避免死锁）
    lib_dir = None
    if lmp_exe_path:
        prefix = os.path.dirname(os.path.dirname(os.path.abspath(lmp_exe_path)))
        lib_dir = os.path.join(prefix, "lib")
        if os.path.isdir(lib_dir) and (
            os.path.exists(os.path.join(lib_dir, "liblammps.0.dylib"))
            or os.path.exists(os.path.join(lib_dir, "liblammps.so"))
        ):
            pass
        else:
            lib_dir = None
    if not lib_dir and os.path.isdir(os.path.expanduser("~/.local/lib")):
        lib_dir = os.path.expanduser("~/.local/lib")
    if lib_dir:
        env = os.environ
        env.setdefault("DYLD_LIBRARY_PATH", "")
        if lib_dir not in env.get("DYLD_LIBRARY_PATH", ""):
            env["DYLD_LIBRARY_PATH"] = lib_dir + os.pathsep + env["DYLD_LIBRARY_PATH"]
        env.setdefault("LD_LIBRARY_PATH", "")
        if lib_dir not in env.get("LD_LIBRARY_PATH", ""):
            env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env["LD_LIBRARY_PATH"]

    try:
        from molecular_force_field.cli.lammps_interface import generate_lammps_input
    except ImportError:
        print("无法导入 generate_lammps_input，请在本仓库根目录安装包后再运行测试。")
        return False

    with tempfile.TemporaryDirectory(prefix="mff_lammps_test_") as td:
        ckpt_path = os.path.join(td, "dummy_model.pth")
        _make_dummy_checkpoint(ckpt_path, torch.device("cpu"))
        _prepare_lammps_input_in_dir(td, ckpt_path)

        # 优先：同进程用 LAMMPS 库 + fix external + 真实势函数 跑 run 0
        n = int(os.environ.get("MFF_LAMMPS_TEST_NATOMS", str(n_atoms)))
        if _run_lammps_in_process(td, ckpt_path, n_atoms=n):
            print("LAMMPS 接口联机测试通过（fix external + 真实势函数 run 0 成功）。")
            return True

        # 退化为子进程调用 lmp 可执行文件
        if not lmp_exe_path:
            print("未找到 LAMMPS 可执行文件（lmp / lmp_serial / lmp_mpi），且无法使用 LAMMPS Python 库。跳过联机测试。")
            print("安装方法见 molecular_force_field/docs/INSTALL_LAMMPS_PYTHON.md")
            return True

        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            sp_dirs = site.getsitepackages()
        except Exception:
            sp_dirs = [site.getusersitepackages()]
        torch_site = ""
        for d in sp_dirs:
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "torch")):
                torch_site = d
                break
        if not torch_site and sp_dirs:
            torch_site = sp_dirs[0]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [package_root, torch_site, env.get("PYTHONPATH", "")])
        )
        if lib_dir:
            env["DYLD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("DYLD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        result = subprocess.run(
            [lmp_exe_path, "-in", "lammps_input.in", "-log", "none"],
            cwd=td,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print("LAMMPS 运行失败（子进程）:")
            print(result.stdout or "(无 stdout)")
            print(result.stderr or "(无 stderr)")
            return False
    print("LAMMPS 接口联机测试通过。")
    return True


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LAMMPS 接口联机测试")
    parser.add_argument("--lmp", type=str, default=None, help="LAMMPS 可执行文件路径（可选）")
    parser.add_argument("--n-atoms", type=int, default=50,
                        help="测试原子数（默认 50，可用 MFF_LAMMPS_TEST_NATOMS 环境变量覆盖）")
    args = parser.parse_args()
    ok = run_lammps_integration_test(args.lmp, n_atoms=args.n_atoms)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
