"""ASE Calculator 联机测试：用 dummy 模型验证 MyE3NNCalculator 能量/力计算。

运行：
    python -m molecular_force_field.test.test_ase_calculator
    python -m molecular_force_field.test.test_ase_calculator --n-atoms 50
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# 限制线程，与 LAMMPS 测试一致
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from molecular_force_field.test.self_test_lammps_potential import _make_dummy_checkpoint
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
from molecular_force_field.evaluation.calculator import MyE3NNCalculator


def _build_ase_atoms(n_atoms: int, box_size: float = 25.0):
    """构建 ASE Atoms：小体系 H-O-H，大体系网格 H/O。"""
    from ase import Atoms
    if n_atoms <= 3:
        symbols = ["H", "O", "H"]
        positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.24, 0.93, 0.0]], dtype=np.float64)
        return Atoms(symbols=symbols[:n_atoms], positions=positions[:n_atoms], pbc=False)
    # 大体系：网格
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
    return Atoms(symbols=symbols, positions=np.array(positions), pbc=False)


def run_ase_test(n_atoms: int = 50, n_steps: int = 10, device: str = "cpu") -> bool:
    """运行 ASE Calculator 测试。"""
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "dummy_model.pth")
        _make_dummy_checkpoint(ckpt_path, torch.device(device))

        # 加载模型
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        dtype = torch.float64
        config = ModelConfig(dtype=dtype)
        if config.atomic_energy_keys is None:
            config.atomic_energy_keys = torch.tensor([1, 8], dtype=torch.long)
            config.atomic_energy_values = torch.tensor([-13.6, -75.0], dtype=dtype)

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
        model.load_state_dict(ckpt["e3trans_state_dict"], strict=True)
        model.eval()

        atomic_energies_dict = {
            int(k): float(v) for k, v in zip(
                config.atomic_energy_keys.tolist(),
                config.atomic_energy_values.tolist(),
            )
        }

        calc = MyE3NNCalculator(
            model, atomic_energies_dict, torch.device(device), max_radius=3.0
        )
        atoms = _build_ase_atoms(n_atoms)
        atoms.calc = calc

        # 单步能量/力
        t0 = time.perf_counter()
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        t1 = time.perf_counter()
        print(f"ASE 单步: {n_atoms} 原子, E={e:.4f} eV, 耗时 {t1-t0:.4f} s")

        # 多步 MD（可选）
        if n_steps > 0 and n_atoms > 3:
            from ase.md.verlet import VelocityVerlet
            from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
            from ase import units
            MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(42))
            dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
            t0 = time.perf_counter()
            dyn.run(n_steps)
            t1 = time.perf_counter()
            print(f"ASE MD {n_steps} 步: 耗时 {t1-t0:.4f} s ({(t1-t0)/n_steps*1000:.1f} ms/step)")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ASE Calculator 联机测试")
    parser.add_argument("--n-atoms", type=int, default=50, help="原子数")
    parser.add_argument("--n-steps", type=int, default=10, help="MD 步数（0 则跳过）")
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    args = parser.parse_args()

    try:
        ok = run_ase_test(n_atoms=args.n_atoms, n_steps=args.n_steps, device=args.device)
    except Exception as e:
        print(f"ASE 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("ASE Calculator 联机测试通过。")
    sys.exit(0)


if __name__ == "__main__":
    main()
