#!/usr/bin/env bash
# 在本地安装 CPU 版 LAMMPS，并启用 Python 接口（供 fix python/invoke、pair_style python 使用）
#
# 用法（二选一）：
#   A) Conda（推荐，最简单）:
#        conda create -n lammps python=3.10 -y && conda activate lammps
#        conda install -c conda-forge lammps
#   B) 本脚本从源码编译（需已安装 CMake、C++17 编译器）:
#        bash scripts/install_lammps_cpu_python.sh [安装目录，默认 $HOME/.local]
#
set -euo pipefail

INSTALL_PREFIX="${1:-$HOME/.local}"
LAMMPS_VERSION="${LAMMPS_VERSION:-stable}"
BUILD_DIR="${BUILD_DIR:-/tmp/lammps_build}"

echo "=== LAMMPS CPU + Python 安装 ==="
echo "安装前缀: $INSTALL_PREFIX"
echo ""

# 检查是否已通过 conda 安装
if command -v lmp &>/dev/null; then
  echo "检测到已安装的 LAMMPS 可执行文件: $(which lmp)"
  lmp -h 2>/dev/null | head -1 || true
  echo "若需从源码重装，请先卸载 conda 中的 lammps 或使用新环境。"
  exit 0
fi

# 从源码编译
echo "从源码编译 LAMMPS（CPU、共享库、PYTHON 包）..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [[ ! -d lammps ]]; then
  echo "正在下载 LAMMPS 源码..."
  if command -v git &>/dev/null; then
    git clone --depth 1 https://github.com/lammps/lammps.git -b stable
  else
    echo "请先安装 git，或从 https://github.com/lammps/lammps/releases 下载源码解压到 $BUILD_DIR/lammps"
    exit 1
  fi
fi

cd lammps
mkdir -p build
cd build

# CPU、共享库、PYTHON 包；串行或 MPI 均可（若未检测到 MPI 会使用 STUBS）
cmake ../cmake \
  -D CMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" \
  -D BUILD_SHARED_LIBS=on \
  -D PKG_PYTHON=on \
  -D CMAKE_BUILD_TYPE=Release

cmake --build . -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
make install

# 安装 Python 模块与共享库到当前 Python 的 site-packages，便于 import lammps 或供 LAMMPS 调用 Python
echo "安装 LAMMPS Python 模块..."
make install-python || true

echo ""
echo "安装完成。将以下内容加入 ~/.zshrc 或 ~/.bashrc 以便 lmp 找到动态库："
echo "  export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
echo "  export DYLD_LIBRARY_PATH=\"$INSTALL_PREFIX/lib:\${DYLD_LIBRARY_PATH:-}\"   # macOS"
echo "  export LD_LIBRARY_PATH=\"$INSTALL_PREFIX/lib:\${LD_LIBRARY_PATH:-}\"         # Linux"
echo "验证："
echo "  lmp -h"
echo "  python -c \"import lammps; lmp = lammps.lammps(); print('Python 接口正常')\""
