#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/install_user_mfftorch_into_lammps.sh /path/to/lammps"
  exit 2
fi

LMP_ROOT="$1"
SRC_DIR="$LMP_ROOT/src"
CMAKE_PKG_DIR="$LMP_ROOT/cmake/Modules/Packages"
CMK_PACKAGES_DIR="$LMP_ROOT/cmake/Packages"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "LAMMPS src directory not found: $SRC_DIR"
  exit 2
fi
if [[ ! -d "$CMAKE_PKG_DIR" ]]; then
  echo "LAMMPS CMake Packages directory not found: $CMAKE_PKG_DIR"
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Copying USER-MFFTORCH source to: $SRC_DIR/USER-MFFTORCH"
mkdir -p "$SRC_DIR/USER-MFFTORCH"
cp -v "$REPO_ROOT/lammps_user_mfftorch/src/USER-MFFTORCH/"* "$SRC_DIR/USER-MFFTORCH/"

echo "Copying CMake package module to: $CMAKE_PKG_DIR/USER-MFFTORCH.cmake"
cp -v "$REPO_ROOT/lammps_user_mfftorch/cmake/Modules/Packages/USER-MFFTORCH.cmake" \
  "$CMAKE_PKG_DIR/USER-MFFTORCH.cmake"

if [[ -d "$CMK_PACKAGES_DIR" ]]; then
  echo "Copying CMake package module to: $CMK_PACKAGES_DIR/USER-MFFTORCH.cmake"
  cp -v "$REPO_ROOT/lammps_user_mfftorch/cmake/Packages/USER-MFFTORCH.cmake" \
    "$CMK_PACKAGES_DIR/USER-MFFTORCH.cmake"
else
  echo "WARN: $CMK_PACKAGES_DIR not found (normal for your LAMMPS version); will only use modules under $CMAKE_PKG_DIR"
fi

echo "DONE. You can now use -D PKG_USER-MFFTORCH=ON in LAMMPS CMake"

