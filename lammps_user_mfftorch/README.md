## 这是什么

这是一套可拷贝进 **LAMMPS 源码树**的自定义包：`USER-MFFTORCH`。

**完整编译与运行指南**：见 [docs/BUILD_AND_RUN.md](docs/BUILD_AND_RUN.md)。

它提供两个 pair style：

- `pair_style mff/torch`：纯 C++ + LibTorch（可先用 CPU neighbor list 跑通）
- `pair_style mff/torch/kk`：Kokkos+CUDA 数据准备 + LibTorch(CUDA) 推理（目标：全链路无 Python、无 Host 往返）

以及一个 compute style：

- `compute ... mff/torch/phys`：读取最近一次 `pair_style mff/torch*` 缓存的物理张量输出

模型文件使用本仓库导出的 **TorchScript core**：`core.pt`（`torch.jit.save`），见 `molecular_force_field/scripts/export_libtorch_core.py`。

## 目录结构（拷贝到 LAMMPS 源码）

把本目录中的：

- `src/USER-MFFTORCH/` → 拷贝到 `LAMMPS/src/USER-MFFTORCH/`
- （你的版本为 22Jul2025）`cmake/Modules/Packages/USER-MFFTORCH.cmake` → 拷贝到 `LAMMPS/cmake/Modules/Packages/USER-MFFTORCH.cmake`

## LAMMPS CMake 编译示例

```bash
cmake -S /path/to/lammps/cmake -B build-lmp \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D CMAKE_PREFIX_PATH="$(python - <<'PY'\nimport torch\nprint(torch.utils.cmake_prefix_path)\nPY\n)"
cmake --build build-lmp -j
```

此外你需要在 `LAMMPS/cmake/CMakeLists.txt` 里做 2 处一次性改动（否则 CMake 不会识别这个新包）：
- 把 `USER-MFFTORCH` 加进 `STANDARD_PACKAGES` 列表
- 把 `USER-MFFTORCH` 加进 `foreach(PKG_WITH_INCL ...)` 的 include 列表（这样会执行 `cmake/Modules/Packages/USER-MFFTORCH.cmake` 来链接 LibTorch）

如果运行时找不到 `libtorch.so` / `libc10.so`，给可执行文件设置 `LD_LIBRARY_PATH`（指向你的 Python venv 里 `torch/lib`）。

## LAMMPS 输入示例

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin

pair_style mff/torch/kk 5.0 cuda
pair_coeff * * /path/to/core.pt H O

velocity all create 300 42
fix 1 all nve
run 100
```

说明：
- `pair_style ... 5.0` 是 cutoff（Angstrom）。
- `pair_coeff * * core.pt H O` 的元素顺序必须与导出时一致（type→元素→Z 映射）。可用 `NULL` 跳过某个 type。

## 物理张量输出

如果导出的 `core.pt` 来自带 `physical_tensor_outputs` 的 `pure-cartesian-ictd` 模型，
那么 `pair_style mff/torch*` 会在每一步缓存固定 schema 的笛卡尔物理张量，可通过：

```lammps
compute mffg all mff/torch/phys global
compute mffgm all mff/torch/phys global/mask
compute mffa all mff/torch/phys atom
compute mffam all mff/torch/phys atom/mask
```

来读取。

- `global`：返回长度为 22 的全局向量，可用于 `thermo_style custom`
- `global/mask`：返回长度为 4 的全局 mask，顺序为 `charge dipole polarizability quadrupole`
- `atom`：返回 `N x 22` 的逐原子数组，可用于 `dump custom`
- `atom/mask`：返回长度为 4 的全局 mask，顺序为 `charge_per_atom dipole_per_atom polarizability_per_atom quadrupole_per_atom`

22 列固定顺序如下：

`[charge, dipole_x, dipole_y, dipole_z, polar_xx, polar_xy, polar_xz, polar_yx, polar_yy, polar_yz, polar_zx, polar_zy, polar_zz, quad_xx, quad_xy, quad_xz, quad_yx, quad_yy, quad_yz, quad_zx, quad_zy, quad_zz]`

若某个物理头未在模型中启用，对应 22 列会填 0，是否启用请看 `mask`。

也支持按名字直接取某个物理量或分量，避免手工记 22 列索引，例如：

```lammps
compute dip all mff/torch/phys global dipole
compute dipx all mff/torch/phys global dipole x
compute pol all mff/torch/phys global polarizability
compute polxx all mff/torch/phys global polarizability xx

compute adip all mff/torch/phys atom dipole
compute adipx all mff/torch/phys atom dipole x
```

- `global charge` / `global dipole x` / `global polarizability xx` 这类单分量会直接返回标量
- `global dipole` 返回 3 维向量，`global polarizability` / `global quadrupole` 返回 9 维向量
- `atom dipole x` 返回逐原子向量，`atom dipole` 返回逐原子 `N x 3` 数组，`atom polarizability` 返回 `N x 9` 数组
- `global/mask dipole` 或 `atom/mask polarizability` 可直接检查某个头是否启用

如果你的模型训练时根本没有配置 `physical_tensor_outputs`，`pair_style mff/torch*` 仍然会正常运行，能量和力链路不受影响。
此时：

- `compute ... mff/torch/phys global` / `atom` 会返回全 0
- `compute ... mff/torch/phys global/mask` / `atom/mask` 会返回全 0

