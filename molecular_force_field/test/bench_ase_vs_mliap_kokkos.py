#!/usr/bin/env python3
"""ASE vs LAMMPS ML-IAP Kokkos 性能对比测试

多组不同体系大小，对比 ASE（PyTorch）与 LAMMPS ML-IAP Kokkos GPU 的单步耗时。

用法:
  python molecular_force_field/scripts/bench_ase_vs_mliap_kokkos.py
  python molecular_force_field/scripts/bench_ase_vs_mliap_kokkos.py --mode spherical-save-cue  # cuEquivariance GPU 加速
  python molecular_force_field/scripts/bench_ase_vs_mliap_kokkos.py --sizes 20 50 100 200 500
  python molecular_force_field/scripts/bench_ase_vs_mliap_kokkos.py --n-steps 20 --device cuda
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_script_dir))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _build_ase_atoms(n_atoms: int, box_size: float = 30.0):
    """构建 ASE Atoms：H/O 网格。"""
    from ase import Atoms
    if n_atoms <= 3:
        symbols = ["H", "O", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]], dtype=np.float64)
        return Atoms(symbols=symbols[:n_atoms], positions=positions[:n_atoms], pbc=True, cell=[box_size]*3)
    n = max(2, int(np.ceil(n_atoms ** (1 / 3))))
    spacing = max(1.2, (box_size - 2) / n)
    symbols, positions = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if len(symbols) >= n_atoms:
                    break
                sym = "H" if (len(symbols) % 3) != 0 else "O"
                symbols.append(sym)
                positions.append([1.0 + i * spacing, 1.0 + j * spacing, 1.0 + k * spacing])
            if len(symbols) >= n_atoms:
                break
        if len(symbols) >= n_atoms:
            break
    atoms = Atoms(symbols=symbols, positions=np.array(positions), pbc=True, cell=[box_size]*3)
    return atoms


def bench_ase(ckpt_path: str, n_atoms: int, n_steps: int, device: str, mode_override: str | None = None) -> float:
    """ASE MD 单步平均耗时 (ms/step)。默认使用 pure_cartesian_ictd_layers_full。"""
    from molecular_force_field.models import E3_TransformerLayer_multi
    from molecular_force_field.models.pure_cartesian_ictd_layers_full import PureCartesianICTDTransformerLayer
    from molecular_force_field.utils.config import ModelConfig
    from molecular_force_field.evaluation.calculator import MyE3NNCalculator
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mode = mode_override if mode_override is not None else ckpt.get("tensor_product_mode", "spherical")
    config = ModelConfig(dtype=torch.float64)
    config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
    config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=torch.float64)

    if mode == "spherical-save":
        from molecular_force_field.models.e3nn_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_channelwise,
        )
        model = E3_TransformerLayer_multi_channelwise(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            device=torch.device(device),
        ).to(device)
    elif mode == "spherical-save-cue":
        from molecular_force_field.models.cue_layers_channelwise import (
            E3_TransformerLayer_multi as E3_TransformerLayer_multi_cue,
        )
        model = E3_TransformerLayer_multi_cue(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            device=torch.device(device),
        ).to(device)
    elif mode == "pure-cartesian-ictd":
        model = PureCartesianICTDTransformerLayer(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            hidden_dim_conv=config.channel_in,
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            num_interaction=2,
            function_type_main=config.function_type,
            lmax=config.lmax,
        internal_compute_dtype=config.internal_compute_dtype,
            device=torch.device(device),
        ).to(device)
    else:
        model = E3_TransformerLayer_multi(
            max_embed_radius=config.max_radius,
            main_max_radius=config.max_radius_main,
            main_number_of_basis=config.number_of_basis_main,
            irreps_input=config.get_irreps_output_conv(),
            irreps_query=config.get_irreps_query_main(),
            irreps_key=config.get_irreps_key_main(),
            irreps_value=config.get_irreps_value_main(),
            irreps_output=config.get_irreps_output_conv_2(),
            irreps_sh=config.get_irreps_sh_transformer(),
            hidden_dim_sh=config.get_hidden_dim_sh(),
            hidden_dim=config.emb_number_main_2,
            channel_in2=config.channel_in2,
            embedding_dim=config.embedding_dim,
            max_atomvalue=config.max_atomvalue,
            output_size=config.output_size,
            embed_size=config.embed_size,
            main_hidden_sizes3=config.main_hidden_sizes3,
            num_layers=config.num_layers,
            function_type_main=config.function_type,
            device=torch.device(device),
        ).to(device)
    is_cue = (mode == "spherical-save-cue")
    model.load_state_dict(ckpt["e3trans_state_dict"], strict=not is_cue)
    model.eval()

    atomic_energies_dict = {1: -13.6, 8: -75.0}
    calc = MyE3NNCalculator(model, atomic_energies_dict, torch.device(device), max_radius=3.0)
    atoms = _build_ase_atoms(n_atoms)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)

    # 预热
    dyn.run(2)
    torch.cuda.synchronize() if device == "cuda" else None

    t0 = time.perf_counter()
    dyn.run(n_steps)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    ms_per_step = (t1 - t0) / n_steps * 1000
    del model, calc, atoms, dyn
    if device == "cuda":
        torch.cuda.empty_cache()
    return ms_per_step


def bench_lammps_mliap_kokkos(model_pt: str, n_atoms: int, n_steps: int, skin: float = 1.0) -> float:
    """LAMMPS ML-IAP Kokkos MD 单步平均耗时 (ms/step)。"""
    try:
        import lammps
        from lammps.mliap import activate_mliappy, load_unified
        try:
            from lammps.mliap import activate_mliappy_kokkos
            use_kokkos = True
        except (ImportError, AttributeError):
            use_kokkos = False
    except ImportError:
        return float("nan")

    # 构建 LAMMPS 输入：create_box + create_atoms random，盒子随体系缩放
    box = max(25.0, 2.5 * (n_atoms ** (1/3)))
    n1 = (2 * n_atoms) // 3  # H
    n2 = n_atoms - n1        # O
    if n2 == 0:
        n2 = 1
        n1 = n_atoms - 1

    lmp = None
    model = None
    try:
        if use_kokkos:
            lmp = lammps.lammps(cmdargs=[
                "-nocite", "-log", "none",
                "-k", "on", "g", "1", "-sf", "kk",
                "-pk", "kokkos", "newton", "on", "neigh", "half",
            ])
            activate_mliappy(lmp)  # 必须先激活 ML-IAP 模块
            activate_mliappy_kokkos(lmp)
        else:
            lmp = lammps.lammps(cmdargs=["-nocite", "-log", "none"])
            activate_mliappy(lmp)

        model = torch.load(model_pt, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        load_unified(model)

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

        # 预热
        lmp.command("run 2")
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        t0 = time.perf_counter()
        lmp.command(f"run {n_steps}")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        return (t1 - t0) / n_steps * 1000  # ms/step
    except Exception:
        return float("nan")
    finally:
        try:
            if lmp is not None:
                lmp.close()
        except Exception:
            pass
        del lmp, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASE vs ML-IAP Kokkos 性能对比")
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100, 200, 500],
                        help="体系大小（原子数）列表，默认最大 500 以避免 ML-IAP OOM")
    parser.add_argument("--n-steps", type=int, default=10, help="每组 MD 步数")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                        help="ASE 使用的设备")
    parser.add_argument("--mode", type=str, default="pure-cartesian-ictd",
                        choices=["pure-cartesian-ictd", "spherical-save", "spherical-save-cue", "spherical"],
                        help="模型模式（默认: pure-cartesian-ictd）。spherical-save-cue 需 GPU+cuEquivariance")
    parser.add_argument("--skin", type=float, default=1.0,
                        help="LAMMPS neighbor skin (Angstrom). metal 默认约 2.0；减小可减少 ghost/邻居数。")
    parser.add_argument("--compare-torchscript", action="store_true",
                        help="额外导出并测试 TorchScript 版本的 ML-IAP（仅 pure-cartesian-ictd 可用）")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，改用 CPU")
        device = "cpu"

    mode_labels = {
        "pure-cartesian-ictd": "pure_cartesian_ictd_layers_full",
        "spherical-save": "e3nn_layers_channelwise",
        "spherical-save-cue": "cue_layers_channelwise (cuEquivariance)",
        "spherical": "e3nn_layers",
    }
    print("=" * 70)
    print("ASE vs LAMMPS ML-IAP Kokkos 性能对比")
    print("=" * 70)
    print(f"模型: {args.mode} ({mode_labels[args.mode]})")
    print(f"ASE 设备: {device}")
    print(f"ML-IAP: Kokkos GPU")
    print(f"LAMMPS neighbor skin: {args.skin}")
    print(f"每组 MD 步数: {args.n_steps}")
    print(f"体系大小: {args.sizes}")
    print()

    with tempfile.TemporaryDirectory() as td:
        from molecular_force_field.test.test_mliap import _make_dummy_checkpoint_for_mode
        from molecular_force_field.interfaces.lammps_mliap import LAMMPS_MLIAP_MFF

        ckpt_path = os.path.join(td, "dummy.pth")
        _make_dummy_checkpoint_for_mode(args.mode, ckpt_path, torch.device("cpu"))
        model_pt = os.path.join(td, "model-mliap.pt")
        mliap_device = "cuda" if torch.cuda.is_available() else "cpu"
        obj = LAMMPS_MLIAP_MFF.from_checkpoint(
            ckpt_path, element_types=["H", "O"], max_radius=3.0,
            atomic_energy_keys=[1, 8], atomic_energy_values=[-13.6, -75.0],
            device=mliap_device,
        )
        try:
            torch.save(obj, model_pt)
            mliap_available = True
        except (AttributeError, TypeError, RuntimeError) as e:
            err_str = str(e).lower()
            if "pickle" in err_str or "can't pickle" in err_str or "cannot pickle" in err_str:
                print(f"注意: {args.mode} 模型无法序列化（cuEquivariance 内部限制），仅运行 ASE 测试")
                mliap_available = False
            else:
                raise

        # Optional TorchScript variant (currently supported for pure-cartesian-ictd only)
        model_pt_ts = None
        mliap_ts_available = False
        if args.compare_torchscript and mliap_available:
            if args.mode != "pure-cartesian-ictd":
                print("注意: --compare-torchscript 目前只支持 pure-cartesian-ictd；已跳过 TorchScript 对比")
            else:
                try:
                    # IMPORTANT: trace TorchScript on CUDA if available. If we trace on CPU,
                    # device moves inside the model can get baked as CPU no-ops, causing
                    # cuda/cpu mismatches at runtime under Kokkos.
                    ts_device = "cuda" if torch.cuda.is_available() else "cpu"
                    obj_ts = LAMMPS_MLIAP_MFF.from_checkpoint(
                        ckpt_path, element_types=["H", "O"], max_radius=3.0,
                        atomic_energy_keys=[1, 8], atomic_energy_values=[-13.6, -75.0],
                        device=ts_device,
                        torchscript=True,
                    )
                    model_pt_ts = os.path.join(td, "model-mliap-torchscript.pt")
                    torch.save(obj_ts, model_pt_ts)
                    mliap_ts_available = True
                except Exception as e:
                    print(f"注意: TorchScript 导出失败，跳过对比。错误: {e}")

        results = []
        for n in args.sizes:
            print(f"  测试 {n} 原子...", end=" ", flush=True)
            if device == "cuda":
                torch.cuda.empty_cache()
            try:
                t_ase = bench_ase(ckpt_path, n, args.n_steps, device, mode_override=args.mode)
            except Exception as e:
                t_ase = float("nan")
                print(f"ASE 失败: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
            if mliap_available:
                try:
                    t_mliap = bench_lammps_mliap_kokkos(model_pt, n, args.n_steps, skin=float(args.skin))
                except Exception as e:
                    t_mliap = float("nan")
                    print(f"ML-IAP 失败: {e}")
            else:
                t_mliap = float("nan")

            t_mliap_ts = float("nan")
            if mliap_ts_available and model_pt_ts is not None:
                try:
                    t_mliap_ts = bench_lammps_mliap_kokkos(model_pt_ts, n, args.n_steps, skin=float(args.skin))
                except Exception as e:
                    t_mliap_ts = float("nan")
                    print(f"ML-IAP(TorchScript) 失败: {e}")

            # Print one-line summary per size
            if args.compare_torchscript and mliap_ts_available:
                if not (np.isnan(t_ase) or np.isnan(t_mliap) or np.isnan(t_mliap_ts)):
                    r_eager = t_ase / t_mliap if t_mliap > 0 else float("nan")
                    r_ts = t_ase / t_mliap_ts if t_mliap_ts > 0 else float("nan")
                    results.append((n, t_ase, t_mliap, t_mliap_ts, r_eager, r_ts))
                    print(
                        f"ASE={t_ase:.1f} ms, ML-IAP(eager)={t_mliap:.1f} ms ({r_eager:.2f}x), "
                        f"ML-IAP(TS)={t_mliap_ts:.1f} ms ({r_ts:.2f}x)"
                    )
                else:
                    results.append((n, t_ase, t_mliap, t_mliap_ts, float("nan"), float("nan")))
                    print("跳过")
            else:
                if not (np.isnan(t_ase) or np.isnan(t_mliap)):
                    ratio = t_ase / t_mliap if t_mliap > 0 else float("nan")
                    results.append((n, t_ase, t_mliap, ratio))
                    print(f"ASE={t_ase:.1f} ms/step, ML-IAP={t_mliap:.1f} ms/step, 比值={ratio:.2f}x")
                else:
                    results.append((n, t_ase, t_mliap, float("nan")))
                    print("跳过")

    # 打印表格
    print()
    print("=" * 70)
    print("性能对比结果")
    print("=" * 70)
    if args.compare_torchscript and mliap_ts_available:
        print(f"{'N_atoms':>10} | {'ASE (ms/step)':>14} | {'ML-IAP eager (ms/step)':>22} | {'ML-IAP TS (ms/step)':>19} | {'ASE/eager':>9} | {'ASE/TS':>7}")
        print("-" * 100)
        for n, t_ase, t_mliap, t_ts, r_eager, r_ts in results:
            ase_s = f"{t_ase:.2f}" if not np.isnan(t_ase) else "N/A"
            eager_s = f"{t_mliap:.2f}" if not np.isnan(t_mliap) else "N/A"
            ts_s = f"{t_ts:.2f}" if not np.isnan(t_ts) else "N/A"
            r1 = f"{r_eager:.2f}x" if not np.isnan(r_eager) else "N/A"
            r2 = f"{r_ts:.2f}x" if not np.isnan(r_ts) else "N/A"
            print(f"{n:>10} | {ase_s:>14} | {eager_s:>22} | {ts_s:>19} | {r1:>9} | {r2:>7}")
        print("=" * 70)
        print()
        print("说明: ASE/ML-IAP > 1 表示 ML-IAP 更快；< 1 表示 ASE 更快。")
        print("注意: TorchScript 仍走 Python 回调；性能瓶颈可能不在模型本身。")
    else:
        print(f"{'N_atoms':>10} | {'ASE (ms/step)':>14} | {'ML-IAP Kokkos (ms/step)':>22} | {'ASE/ML-IAP':>10}")
        print("-" * 70)
        for n, t_ase, t_mliap, ratio in results:
            ase_s = f"{t_ase:.2f}" if not np.isnan(t_ase) else "N/A"
            mliap_s = f"{t_mliap:.2f}" if not np.isnan(t_mliap) else "N/A"
            ratio_s = f"{ratio:.2f}x" if not np.isnan(ratio) else "N/A"
            print(f"{n:>10} | {ase_s:>14} | {mliap_s:>22} | {ratio_s:>10}")
        print("=" * 70)
        print()
        print("说明: ASE/ML-IAP > 1 表示 ML-IAP Kokkos 更快；< 1 表示 ASE 更快。")
        print("注意: ML-IAP 大体系易 OOM，可减小 --sizes；ASE 大体系可设 --device cpu 避免显存不足。")


if __name__ == "__main__":
    main()
