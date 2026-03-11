# LAMMPS + Kokkos + LibTorch 完整编译与运行指南

本文档描述如何从零编译带 **Kokkos GPU** 与 **USER-MFFTORCH**（LibTorch 势函数）的 LAMMPS，并完成端到端 MD 运行。

## 目录

1. [前置条件](#1-前置条件)
2. [获取 LAMMPS 源码](#2-获取-lammps-源码)
3. [安装 USER-MFFTORCH](#3-安装-user-mfftorch-到-lammps-源码树)
4. [修改 CMakeLists.txt](#4-修改-lammps-cmakeliststxt一次性)
5. [CMake 配置](#5-cmake-配置)
6. [编译](#6-编译)
7. [导出 core.pt](#7-导出-corepttorchscript-模型)
8. [运行 LAMMPS](#8-运行-lammps)
9. [一键测试脚本](#9-一键测试脚本)
10. [故障排查](#10-故障排查)

---

## 1. 前置条件

### 1.1 系统与工具

| 依赖 | 要求 |
|------|------|
| 操作系统 | Linux（推荐，用于 CUDA） |
| CMake | ≥ 3.20 |
| C++ 编译器 | GCC 7+ 或 Clang，支持 C++17 |
| CUDA | 11+（与 PyTorch/LibTorch 版本匹配） |
| Python | 3.8+（用于导出 `core.pt`） |

### 1.2 Python 环境

- 安装 PyTorch（含 CUDA 支持，与 GPU 驱动版本匹配）：
  ```bash
  pip install torch  # 或根据 CUDA 版本选择 torch+cu118 等
  ```
- 安装本仓库：
  ```bash
  pip install -e .  # 在 rebuild 仓库根目录
  ```

### 1.3 LibTorch 路径

USER-MFFTORCH 通过 `find_package(Torch)` 查找 LibTorch。推荐使用 **Python 内置的 LibTorch**：

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

输出示例：`/path/to/python/site-packages/torch/share/cmake/Torch`。将该路径传给 CMake 的 `CMAKE_PREFIX_PATH`。

---

## 2. 获取 LAMMPS 源码

```bash
# 方式 A：下载官方源码
wget https://download.lammps.org/tars/lammps-22Jul2025.tar.gz
tar xzf lammps-22Jul2025.tar.gz
cd lammps-22Jul2025

# 方式 B：Git 克隆
git clone https://github.com/lammps/lammps.git
cd lammps
git checkout 22Jul2025  # 或 develop 等
```

---

## 3. 安装 USER-MFFTORCH 到 LAMMPS 源码树

在 **rebuild 仓库根目录** 执行：

```bash
bash scripts/install_user_mfftorch_into_lammps.sh /path/to/lammps
```

例如：

```bash
bash scripts/install_user_mfftorch_into_lammps.sh /root/lammps-22Jul2025
```

脚本会：

- 拷贝 `src/USER-MFFTORCH/` → `LAMMPS/src/USER-MFFTORCH/`
- 拷贝 `USER-MFFTORCH.cmake` → `LAMMPS/cmake/Modules/Packages/`（或 `cmake/Packages/`，视版本而定）

---

## 4. 修改 LAMMPS CMakeLists.txt（一次性）

LAMMPS 默认不识别 `USER-MFFTORCH`，需手动修改 `LAMMPS/cmake/CMakeLists.txt`。

### 4.1 添加 USER-MFFTORCH 到 STANDARD_PACKAGES

找到 `set(STANDARD_PACKAGES ...)`，在列表末尾添加 `USER-MFFTORCH`：

```cmake
set(STANDARD_PACKAGES
   ...
   VTK
   YAFF
   USER-MFFTORCH)  # 添加此行
```

### 4.2 添加 USER-MFFTORCH 到 PKG_WITH_INCL

找到 `foreach(PKG_WITH_INCL ...)` 的 include 列表（用于链接外部库的包，如 GRAPHICS、PYTHON、ML-IAP 等），在其中加入 `USER-MFFTORCH`。

**方式 A：加入现有 foreach 列表**

```cmake
foreach(PKG_WITH_INCL GRAPHICS KSPACE PYTHON ML-IAP ... USER-MFFTORCH)
  if(PKG_${PKG_WITH_INCL})
    include(Packages/${PKG_WITH_INCL})
  endif()
endforeach()
```

**方式 B：单独添加（若找不到合适列表）**

在 `foreach(PKG_WITH_INCL ...)` 附近添加：

```cmake
if(PKG_USER-MFFTORCH)
  include(Packages/USER-MFFTORCH)
  # 若报错找不到文件，改为：include(Modules/Packages/USER-MFFTORCH)
endif()
```

> `Packages/` 对应 `cmake/Packages/USER-MFFTORCH.cmake`；`Modules/Packages/` 对应 `cmake/Modules/Packages/USER-MFFTORCH.cmake`。安装脚本会拷贝到两个位置，按你 LAMMPS 版本的实际 include 路径选择其一。

---

## 5. CMake 配置

### 5.1 获取 LibTorch 路径

```bash
LIBTORCH_PREFIX=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
```

### 5.2 选择 GPU 架构

根据你的 GPU 选择 `Kokkos_ARCH_XXX`：

| GPU | Kokkos_ARCH |
|-----|-------------|
| NVIDIA A100 | AMPERE80 |
| NVIDIA A30 / RTX 30 系列 | AMPERE86 |
| NVIDIA V100 | VOLTA70 |
| NVIDIA T4 | TURING75 |
| AMD GPU | 参见 Kokkos 文档 |

### 5.3 编译选项：Virial/应力计算

| 选项 | 默认 | 说明 |
|------|------|------|
| `MFF_ENABLE_VIRIAL` | OFF | 设为 ON 时，在 GPU 上计算 virial，可输出正确的压力/应力张量 |

- **默认（OFF）**：不计算 virial，`thermo_style` 中的 `press` 等只有动能贡献，数值不完整；吞吐量最高。
- **ON**：在 GPU 上用 Kokkos 计算 fdotr virial，只拷贝 6 个 double 回 CPU，**几乎无性能损失**；`press`、`pxx`、`pyy`、`pzz`、`pxy`、`pxz`、`pyz` 输出正确。

### 5.4 配置命令

**不启用 virial（默认，最高吞吐）：**

```bash
cd /path/to/lammps

cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D CMAKE_BUILD_TYPE=Release
```

**启用 virial（压力/应力正确）：**

```bash
cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON \
  -D CMAKE_PREFIX_PATH="$LIBTORCH_PREFIX" \
  -D CMAKE_BUILD_TYPE=Release
```

**若 LibTorch 未被自动找到**，可显式指定：

```bash
cmake -S cmake -B build-mfftorch \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_AMPERE86=ON \
  -D PKG_USER-MFFTORCH=ON \
  -D MFF_ENABLE_VIRIAL=ON \
  -D CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -D CMAKE_BUILD_TYPE=Release
```

---

## 6. 编译

```bash
cmake --build build-mfftorch -j
```

编译成功后，可执行文件位于：

```
build-mfftorch/lmp
```

---

## 7. 导出 core.pt（TorchScript 模型）

LAMMPS 的 `pair_style mff/torch` 需要 `core.pt` 文件。可用两种方式：

### 7.1 方式 A：从真实 checkpoint 导出

```bash
mff-export-core \
  --checkpoint /path/to/model.pth \
  --elements H O \
  --device cuda \
  --dtype float32 \
  --e0-csv /path/to/fitted_E0.csv \
  --out core.pt
```

**--mode 参数**：支持 `pure-cartesian-ictd`、`pure-cartesian-ictd-save`、`spherical-save-cue`。不指定时自动从 checkpoint 读取。若 checkpoint 中保存了 `tensor_product_mode`（mff-train 会自动保存），则无需手动指定。

**spherical-save-cue 导出（方案 A，便携版）**：
- 默认导出为纯 PyTorch 实现，`core.pt` **无需 cuEquivariance 运行时**，可在任意 LibTorch 环境运行。
- 导出时使用 `force_naive`，将 cuEquivariance 自定义 ops 替换为纯 PyTorch 等价实现。

**max-radius 如何确定**：
- **新训练的 checkpoint**（mff-train 会保存 max_radius）：脚本会自动从 checkpoint 读取，无需手动指定。
- **旧 checkpoint 或未保存**：必须与训练时 `mff-train --max-radius` 一致（默认 5.0），且与 LAMMPS `pair_style mff/torch CUTOFF` 的 cutoff 一致。

**E0 默认行为**：
- `mff-export-core` 现在默认会把 E0 一起嵌入导出的 `core.pt`
- 若显式传 `--e0-csv`，则优先使用该文件
- 若不传 `--e0-csv`，新 checkpoint 会优先使用 checkpoint 中保存的 `atomic_energy_keys/atomic_energy_values`
- 若是老 checkpoint 且未保存 E0，则回退到本地 `fitted_E0.csv`
- 只有显式传 `--no-embed-e0` 时，才导出不带 E0 的纯网络能量

### 7.2 方式 B：生成 dummy 模型（用于测试）

```bash
python - <<'PY'
import torch
from molecular_force_field.interfaces.self_test_lammps_potential import _make_dummy_checkpoint_pure_cartesian_ictd
_make_dummy_checkpoint_pure_cartesian_ictd("dummy.pth", device=torch.device("cpu"))
print("dummy.pth 已生成")
PY

python molecular_force_field/scripts/export_libtorch_core.py \
  --checkpoint dummy.pth \
  --elements H O \
  --device cuda \
  --max-radius 5.0 \
  --out core.pt
```

---

## 8. 运行 LAMMPS

### 8.1 设置环境变量（LibTorch 动态库）

```bash
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"
```

### 8.2 最小 LAMMPS 输入示例

创建 `in.mfftorch`：

```lammps
units metal
atom_style atomic
boundary p p p

region box block 0 40 0 40 0 40
create_box 2 box
create_atoms 1 random 200 12345 box
create_atoms 2 random 100 12346 box
mass 1 1.008
mass 2 15.999

neighbor 1.0 bin

pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

compute mffg all mff/torch/phys global
compute mffa all mff/torch/phys atom

velocity all create 300 42
fix 1 all nve
thermo 20
thermo_style custom step temp pe c_mffg[2] c_mffg[3] c_mffg[4]
dump 1 all custom 20 dump.mff id type x y z c_mffa[1] c_mffa[2] c_mffa[3] c_mffa[4]
run 200
```

若 `core.pt` 来自带外场架构的 `pure-cartesian-ictd` checkpoint，可在运行时通过 LAMMPS equal-style 变量传入 rank-1 外场：

```lammps
variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01

pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/core.pt H O
```

说明：
- `field v_Ex v_Ey v_Ez` 会在每个 force call 重新求值，可用于时间相关外场。
- 也支持 rank-2 外场：
  - `field9`：全量 `3x3`，行主序 `xx xy xz yx yy yz zx zy zz`
  - `field6`：对称 `3x3` 简写，顺序 `xx yy zz xy xz yz`
- 若 `core.pt` 需要外场但未提供 `field ...`，LAMMPS 初始化时会报错；反之，普通 `core.pt` 也不能搭配 `field ...` 使用。

rank-2 示例：

```lammps
variable Txx equal 1.0
variable Txy equal 0.0
variable Txz equal 0.0
variable Tyx equal 0.0
variable Tyy equal 1.0
variable Tyz equal 0.0
variable Tzx equal 0.0
variable Tzy equal 0.0
variable Tzz equal 1.0

pair_style mff/torch 5.0 cuda field9 v_Txx v_Txy v_Txz v_Tyx v_Tyy v_Tyz v_Tzx v_Tzy v_Tzz
pair_coeff * * /path/to/core.pt H O
```

### 8.3 输出物理张量

若 `core.pt` 是带 `physical_tensor_outputs` 的 `pure-cartesian-ictd` 模型，
可通过 `compute mff/torch/phys` 读取最近一次 `pair_style mff/torch*` 缓存的笛卡尔物理张量：

```lammps
compute mffg all mff/torch/phys global
compute mffgm all mff/torch/phys global/mask
compute mffa all mff/torch/phys atom
compute mffam all mff/torch/phys atom/mask

compute mffd all mff/torch/phys global dipole
compute mffdx all mff/torch/phys global dipole x
compute mffp all mff/torch/phys global polarizability
compute mffpxx all mff/torch/phys global polarizability xx

compute mffad all mff/torch/phys atom dipole
compute mffadx all mff/torch/phys atom dipole x
```

模式说明：

- `global`：返回 22 维全局向量，可用于 `thermo_style custom`
- `global/mask`：返回 4 维全局 mask，顺序是 `charge dipole polarizability quadrupole`
- `atom`：返回 `N x 22` 逐原子数组，可用于 `dump custom`
- `atom/mask`：返回 4 维全局 mask，顺序是 `charge_per_atom dipole_per_atom polarizability_per_atom quadrupole_per_atom`

也支持按名字直接读取某个物理量：

- `global charge`：全局标量
- `global dipole`：3 维全局向量；`global dipole x`：全局标量
- `global polarizability` / `global quadrupole`：9 维全局向量；`... xx` 等分量形式返回全局标量
- `atom charge`：逐原子向量
- `atom dipole`：逐原子 `N x 3` 数组；`atom dipole x`：逐原子向量
- `atom polarizability` / `atom quadrupole`：逐原子 `N x 9` 数组；`... xx` 等分量形式返回逐原子向量
- `global/mask dipole`、`atom/mask polarizability`：直接检查单个 head 是否启用

22 列固定顺序如下：

`charge, dipole_x, dipole_y, dipole_z, polar_xx, polar_xy, polar_xz, polar_yx, polar_yy, polar_yz, polar_zx, polar_zy, polar_zz, quad_xx, quad_xy, quad_xz, quad_yx, quad_yy, quad_yz, quad_zx, quad_zy, quad_zz`

如果某个头没有在模型中启用，对应列会返回 0，需要配合 `mask` 判断哪些块有效。
另外，该 compute 读取的是当前 timestep 的 pair 缓存；若当前步还未进行 pair 计算，请先执行 `run` 或 `run 0`。

如果模型训练时没有配置 `physical_tensor_outputs`，`pair_style mff/torch*` 仍可正常输出能量和力。
此时 `compute mff/torch/phys` 不会报错，但：

- `global` / `atom` 返回全 0
- `global/mask` / `atom/mask` 也返回全 0

### 8.4 输出压力/应力（virial）

若编译时启用了 `MFF_ENABLE_VIRIAL=ON`，可在 `thermo_style` 中加入压力与应力分量：

```lammps
thermo_style custom step temp pe ke etotal press pxx pyy pzz pxy pxz pyz
thermo 20
```

| 字段 | 含义 |
|------|------|
| `press` | 总压力（标量） |
| `pxx pyy pzz` | 压力张量对角分量 |
| `pxy pxz pyz` | 压力张量非对角分量（剪切应力） |

未启用 virial 时，`press` 等只有动能贡献，数值不完整。

### 8.5 使用 Kokkos GPU 运行

```bash
/path/to/lammps/build-mfftorch/lmp \
  -k on g 1 \
  -sf kk \
  -pk kokkos newton off neigh full \
  -in in.mfftorch
```

说明：

- `-k on g 1`：启用 Kokkos，使用 1 块 GPU
- `-sf kk`：将 `pair_style mff/torch` 映射到 Kokkos 变体 `mff/torch/kk`
- `-pk kokkos newton off neigh full`：Kokkos 使用 `neigh full` 时必须 `newton off`

### 8.6 仅 CPU 运行（无 Kokkos）

```bash
/path/to/lammps/build-mfftorch/lmp -in in.mfftorch
```

---

## 9. 一键测试脚本

### 9.1 快速 smoke 测试（dummy 模型）

```bash
# 在 rebuild 仓库根目录
bash lammps_user_mfftorch/examples/run_smoke_mfftorch.sh /path/to/lmp cuda
```

### 9.2 完整 GPU 测试（含 dummy 或真实模型）

```bash
# 使用 dummy 模型（pure-cartesian-ictd）
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --elements H O \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200

# 使用 dummy 模型（spherical-save-cue，需 cuEquivariance）
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-cue \
  --elements H O \
  --cutoff 5.0 \
  --steps 200

# 使用真实模型
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --pth /path/to/model.pth \
  --elements H O \
  --e0-csv /path/to/fitted_E0.csv \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200
```

### 9.3 Feature-space FFT 训练 smoke（单卡 / 多卡）

```bash
# 单卡：同时测试 ICTD + spherical-save-cue
bash molecular_force_field/test/run_feature_fft_train_smoketest.sh \
  --mode both \
  --device cuda \
  --dtype float32
```

```bash
# 多卡：2 GPU DDP dry run
bash molecular_force_field/test/run_feature_fft_train_smoketest.sh \
  --mode both \
  --device cuda \
  --dtype float32 \
  --n-gpu 2
```

说明：
- 脚本会自动生成非退化的 periodic toy 数据集并先完成预处理
- 然后运行带 `feature_spectral_mode=fft` 的 `mff-train` dry run
- `--mode ictd|cue|both` 可选择后端
- `--n-gpu > 1` 时通过 `mff-train --n-gpu N` 自动触发 DDP

### 9.4 Feature-space FFT 部署 smoke（单卡 / MPI）

```bash
# 单卡 orthogonal PBC smoke
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft \
  --elements H \
  --dtype float32 \
  --cutoff 5.0
```

```bash
# 双卡 MPI：1 rank 对应 1 GPU
CUDA_VISIBLE_DEVICES=0,1 \
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft \
  --elements H \
  --dtype float32 \
  --cutoff 5.0 \
  --gpu 1 \
  --np 2
```

```bash
# 三斜盒 + MPI
CUDA_VISIBLE_DEVICES=0,1 \
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft-triclinic \
  --elements H \
  --dtype float32 \
  --cutoff 5.0 \
  --gpu 1 \
  --np 2
```

```bash
# 2D slab smoke: boundary p p f，仅验证横向周期等价
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft-slab \
  --elements H \
  --dtype float32 \
  --cutoff 5.0
```

```bash
# 2D slab z-open sanity：验证 z 方向没有被误当成周期方向
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft-slab-z-open \
  --elements H \
  --dtype float32 \
  --cutoff 5.0
```

```bash
# np=1 vs np=2 consistency sanity
CUDA_VISIBLE_DEVICES=0,1 \
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft-mpi-consistency \
  --elements H \
  --dtype float32 \
  --cutoff 5.0 \
  --gpu 1 \
  --np 2
```

```bash
# 吞吐量对比：baseline vs feature-FFT / reciprocal path
CUDA_VISIBLE_DEVICES=0,1 \
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lammps/build-mfftorch/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft \
  --compare-throughput \
  --elements H O \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200 \
  --gpu 1 \
  --np 2
```

说明：
- 当前 reciprocal solver 已经是“distributed-contract + correctness-first”的首版：去掉了原子全量 gather，默认走 mesh-reduce / slab-distributed FFT 路径
- 这还不是最终的全程 GPU-resident distributed FFT 实现
- 推荐模式是 `--gpu 1 --np N`，即每个 MPI rank 绑定一张 GPU
- 多 MPI rank 下如果没有 GPU-aware MPI，通信阶段仍可能经过 host staging
- 当前版本已经把 `phi/gx/gy/gz` 合并进一条 batched reciprocal field 路径，减少了重复 transpose / Allgather 次数
- 如果启用 `compute mff/torch/phys`，physical tensor 会按需缓存到 host，这会额外引入输出搬运开销
- `--compare-throughput` 会自动在相同随机体系上跑两次：baseline 与启用长程/feature-FFT 的版本，并输出 `loop time` / `steps/s` / slowdown 对比
- 当前 `boundary p p f` reciprocal 路径已经单独走 `slab_2d` backend：它会在非周期方向使用 vacuum padding，再做正确性优先的频域求解
- `--test-pbc-slab` / `--test-feature-spectral-fft-slab` 只检查 `boundary p p f` 下横向跨边界对与盒内等价对
- `--test-pbc-slab-z-open` / `--test-feature-spectral-fft-slab-z-open` 进一步检查远距 z 对不会被误当成最小像近邻
- 这仍然是 correctness-first 的首版 slab backend，不应等同于最终高性能/高精度 slab Green's function 实现

---

## 10. 故障排查

### 10.1 `torch/torch.h: No such file or directory`

- 原因：LibTorch 未被找到。
- 解决：确保 `CMAKE_PREFIX_PATH` 包含 `torch.utils.cmake_prefix_path` 的路径，且 `USER-MFFTORCH` 已加入 `PKG_WITH_INCL` 的 include 列表。

### 10.2 `Unrecognized pair style 'mff/torch'`

- 原因：USER-MFFTORCH 未正确编译或未启用。
- 解决：确认 `-D PKG_USER-MFFTORCH=ON`，且 `USER-MFFTORCH` 已加入 `STANDARD_PACKAGES`，`src/USER-MFFTORCH/` 存在且包含 `pair_mff_torch.cpp` 等文件。

### 10.3 `error while loading shared libraries: libtorch.so`

- 原因：运行时找不到 LibTorch 动态库。
- 解决：设置 `LD_LIBRARY_PATH`（见 8.1 节）。

### 10.4 `Must use 'newton off' with KOKKOS package option 'neigh full'`

- 原因：Kokkos 配置要求。
- 解决：在 `lmp` 命令行加上 `-pk kokkos newton off neigh full`。

### 10.5 `Kokkos_ARCH` 与 GPU 不匹配

- 现象：运行时报错或 JIT 编译耗时很长。
- 解决：根据 GPU 型号选择正确的 `Kokkos_ARCH_XXX`（见 5.2 节）。

### 10.6 能量/温度恒不变

- 若使用 dummy 模型，输出常数为预期行为（dummy 模型梯度近似为 0）。
- 使用真实训练模型后，能量和温度会随时间变化。

### 10.7 需要压力/应力但未启用 virial

- 现象：`thermo_style` 含 `press` 时，数值只有动能贡献，不完整。
- 解决：用 `-D MFF_ENABLE_VIRIAL=ON` 重新配置并编译，virial 在 GPU 上计算，几乎无性能损失。

---

## 11. 附录：目录结构速览

```
rebuild/
├── lammps_user_mfftorch/
│   ├── src/USER-MFFTORCH/          # 源码
│   ├── cmake/Modules/Packages/USER-MFFTORCH.cmake
│   ├── cmake/Packages/USER-MFFTORCH.cmake  # 部分版本
│   ├── examples/
│   └── docs/BUILD_AND_RUN.md       # 本文档
├── molecular_force_field/
│   └── scripts/
│       ├── export_libtorch_core.py  # 导出 core.pt
│       └── run_gpu_lammps_with_corept.sh
└── scripts/
    └── install_user_mfftorch_into_lammps.sh
```
