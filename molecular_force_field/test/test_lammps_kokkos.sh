#!/bin/bash
# 测试 LAMMPS Kokkos GPU 安装
# 用法: bash test_lammps_kokkos.sh
# 或在 GPU 机器上: chmod +x test_lammps_kokkos.sh && ./test_lammps_kokkos.sh
#
# macOS 若 import lammps 失败，先设置:
#   export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib:$DYLD_LIBRARY_PATH"

# 不设 set -e，以便各步骤独立报告

echo "=========================================="
echo "LAMMPS Kokkos GPU 安装测试"
echo "=========================================="

# 1. 检查 lmp 是否存在
echo ""
echo "--- 1. 检查 lmp 可执行文件 ---"
if command -v lmp &>/dev/null; then
    echo "PASS: lmp 已找到: $(which lmp)"
    lmp -h 2>&1 | head -3
else
    echo "FAIL: 未找到 lmp，请将 LAMMPS build 目录加入 PATH"
    exit 1
fi

# 2. 检查 Python lammps 模块
echo ""
echo "--- 2. 检查 Python lammps 模块 ---"
if python3 -c "import lammps; print('OK')" 2>/dev/null; then
    echo "PASS: lammps 模块可导入"
else
    echo "FAIL: 无法导入 lammps"
    echo "  请确保已执行 make install-python，或设置 PYTHONPATH"
    exit 1
fi

# 3. 检查 CUDA
echo ""
echo "--- 3. 检查 CUDA ---"
if command -v nvidia-smi &>/dev/null; then
    echo "PASS: nvidia-smi 可用"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
else
    echo "WARN: nvidia-smi 不可用（可能非 GPU 环境）"
fi

# 4. 运行 Kokkos 单步测试
echo ""
echo "--- 4. Kokkos GPU 单步测试 ---"
TMP_DIR=$(mktemp -d)
TMP_DATA="$TMP_DIR/test.data"
TMP_IN="$TMP_DIR/test.in"
TMP_LOG="$TMP_DIR/test.log"

cat > "$TMP_DATA" << 'DATA'
Lattice spacing in x,y,z = 1 1 1
4 atoms
2 atom types

0.0 2.0 xlo xhi
0.0 2.0 ylo yhi
0.0 2.0 zlo zhi

Masses
1 1.008
2 15.999

Atoms
1 1 0.0 0.0 0.0
2 2 1.0 0.0 0.0
3 1 0.0 1.0 0.0
4 2 1.0 1.0 0.0
DATA

cat > "$TMP_IN" << 'INPUT'
units           metal
atom_style      atomic
boundary        p p p
read_data       test.data
run             0
INPUT

cd "$TMP_DIR"
if lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in test.in -log test.log 2>&1; then
    echo "PASS: Kokkos 单步运行成功"
else
    echo "FAIL: Kokkos 运行失败"
    cat test.log 2>/dev/null || true
fi
cd - >/dev/null
rm -rf "$TMP_DIR"

# 5. 检查 ML-IAP 相关模块
echo ""
echo "--- 5. 检查 ML-IAP 模块 ---"
if python3 -c "
from lammps.mliap import activate_mliappy, load_unified
print('PASS: activate_mliappy, load_unified 可用')
" 2>/dev/null; then
    :
else
    echo "WARN: lammps.mliap 不可用（标准 ML-IAP）"
fi

# 检查 Kokkos 专用接口
if python3 -c "
from lammps.mliap import activate_mliappy_kokkos
print('PASS: activate_mliappy_kokkos 可用')
" 2>/dev/null; then
    echo "PASS: ML-IAP-Kokkos 接口可用"
else
    echo "WARN: activate_mliappy_kokkos 不可用（需 LAMMPS 2025.09+ 或 Kokkos 版）"
fi

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
