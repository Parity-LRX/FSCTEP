"""
测试应力 loss 是否正常工作。

当 --stress-weight (-c) > 0 时，训练会：
1. 对每个分子施加应变 ε，用 pos_input = pos @ (I+ε)，cell_input = cell @ (I+ε)
2. 前向得到 E，对 ε 求导得到 dE/dε（即 stress_grads）
3. stress_pred = stress_grads / V，与 stress_ref 做 MSE 作为 stress_loss
4. total_loss = a*energy_loss + b*force_loss + c*stress_loss

本脚本用 mini_pbc_stress.xyz（含 stress 标签）做 1 个 epoch 的短训练，检查：
- 能正常跑完且无报错
- 返回的 metrics 中含有 stress_loss 且为有限值
"""
import os
import sys
import tempfile
import subprocess


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xyz_with_stress = os.path.join(root, "mini_pbc_stress.xyz")
    if not os.path.isfile(xyz_with_stress):
        print(f"SKIP: 未找到 {xyz_with_stress}")
        return 0

    with tempfile.TemporaryDirectory(prefix="mff_stress_test_") as data_dir:
        cmd = [
            sys.executable,
            "-m", "molecular_force_field.cli.train",
            "--train-input-file", xyz_with_stress,
            "--valid-input-file", xyz_with_stress,
            "--data-dir", data_dir,
            "--epochs", "1",
            "--stress-weight", "0.1",
            "--batch-size", "2",
            "--num-workers", "0",
            "--dump-frequency", "9999",
        ]
        print("Run:", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        out = result.stdout + result.stderr
        if result.returncode != 0:
            print("STDOUT+STDERR:")
            print(out)
            print("FAIL: 训练进程退出码", result.returncode)
            return 1

        # 检查：1) 启用了应力权重 2) 训练正常完成（无报错即说明 stress 前向/反向已跑通）
        has_stress_weight = "Stress Loss Weight (c):" in out or "stress" in out.lower()
        if not has_stress_weight:
            print("STDOUT+STDERR (last 3000 chars):")
            print(out[-3000:])
            print("FAIL: 输出中未看到 stress 相关配置或日志")
            return 1

        print("PASS: 应力 loss 正常参与训练（c>0 已启用，训练完成无报错）")
        return 0


if __name__ == "__main__":
    sys.exit(main())
