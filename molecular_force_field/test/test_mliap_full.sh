#!/bin/bash
# ML-IAP 完整测试脚本
# 在 GPU 机器上安装 Kokkos LAMMPS 后运行，验证 ML-IAP 全流程
#
# 用法:
#   cd /path/to/rebuild
#   bash molecular_force_field/scripts/test_mliap_full.sh
#   bash molecular_force_field/scripts/test_mliap_full.sh --kokkos   # 含 Kokkos GPU 测试
#
# macOS 若 import lammps 失败，先设置:
#   export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib:$DYLD_LIBRARY_PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python molecular_force_field/scripts/test_mliap_full.py "$@"
