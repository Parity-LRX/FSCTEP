# LAMMPS 接口使用指南

本文档介绍如何将 **FusedEquiTensorPot** 模型集成到 LAMMPS 中进行分子动力学模拟。

## 概述

FusedEquiTensorPot 提供三种 LAMMPS 集成方式：

| 方式 | 速度 | 要求 | 适用场景 |
|------|------|------|----------|
| **USER-MFFTORCH（纯 C++）** | 最快，无 Python/GIL | LAMMPS 编译时开启 KOKKOS + USER-MFFTORCH | HPC、超算、生产部署 |
| **ML-IAP unified** | 较快（比 fix external 约 1.7x） | LAMMPS 编译时开启 ML-IAP | 推荐，支持 GPU 加速 |
| **fix external / pair_style python** | 较慢 | 标准 LAMMPS + Python | 快速验证、无 ML-IAP 时 |

- **USER-MFFTORCH**：模型导出为 TorchScript `core.pt`，LAMMPS 用 LibTorch C++ API 直接加载，**运行时完全不需要 Python**。
- **ML-IAP / fix external**：需 Python 驱动或 Python 解释器，自动处理模型加载、邻居列表、能量/力计算、单位转换。

---

## 方式一：USER-MFFTORCH 纯 C++ 接口（HPC 推荐）

**USER-MFFTORCH** 是自定义 LAMMPS 包，提供 `pair_style mff/torch`，在 C++ 侧用 LibTorch 加载 TorchScript 模型，**无需 Python 解释器**，适合超算与生产部署。

**完整编译与运行流程**见：[lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)

### 快速步骤

1. **导出 core.pt**（需 Python，一次性）：
   ```bash
   python -m molecular_force_field.cli.export_libtorch_core \
     --checkpoint model.pth --elements H O --device cuda \
     --e0-csv fitted_E0.csv --out core.pt
   ```

2. **编译 LAMMPS**：启用 `PKG_KOKKOS`、`PKG_USER-MFFTORCH`，详见 BUILD_AND_RUN.md。

3. **运行**（纯 LAMMPS，无 Python）：
   ```bash
   lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
   ```

**LAMMPS 输入示例**：
```lammps
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O
```

**运行时外场示例**（当前仅支持 `pure-cartesian-ictd` 的 rank-1 外场，即 3 分量电场）：
```lammps
variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01

pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/core.pt H O
```

- `field v_Ex v_Ey v_Ez` 会在每个力计算步重新求值，因此可配合 LAMMPS equal-style 变量实现随时间变化的外场。
- 若 `core.pt` 导出自带外场架构的 checkpoint，则必须提供 `field ...`；若模型不带外场架构，则不能传 `field ...`。

**rank-2 运行时外场示例**：

全量 `3×3` 二阶张量，按行主序 `xx xy xz yx yy yz zx zy zz`：
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

若你的 rank-2 外场本身是对称张量，可用 6 分量简写，顺序为 `xx yy zz xy xz yz`：
```lammps
variable Txx equal 1.0
variable Tyy equal 1.0
variable Tzz equal 1.0
variable Txy equal 0.0
variable Txz equal 0.0
variable Tyz equal 0.0

pair_style mff/torch 5.0 cuda field6 v_Txx v_Tyy v_Tzz v_Txy v_Txz v_Tyz
pair_coeff * * /path/to/core.pt H O
```

- `field9` 适合一般 rank-2 外场，例如电场梯度张量或任意二阶外参。
- `field6` 适合对称 rank-2 张量，例如对称电场梯度近似、应变张量等。

**模型限制**：目前支持 `pure-cartesian-ictd` 系列和 `spherical-save-cue` 模型。元素顺序、cutoff 需与导出时一致。

**spherical-save-cue 导出说明**（方案 A，便携版）：
- 默认导出为**纯 PyTorch 实现**（`force_naive`），`core.pt` 不依赖 cuEquivariance 自定义 ops，可在任意 LibTorch 环境运行。
- 导出命令示例：
  ```bash
  python -m molecular_force_field.cli.export_libtorch_core \
    --checkpoint model.pth --elements H O \
    --e0-csv fitted_E0.csv --out core.pt
  ```
- 一键测试（dummy 模型）：
  ```bash
  bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
    --lmp /path/to/lmp --dummy-cue --elements H O --cutoff 5.0 --steps 200
  ```

---

## 方式二：ML-IAP unified 接口

ML-IAP 是 LAMMPS 的机器学习势接口，比 fix external 更快，且支持 Kokkos GPU。LAMMPS 需编译时开启 `PKG_ML-IAP=ON`、`MLIAP_ENABLE_PYTHON=ON`，详见 `molecular_force_field/docs/INSTALL_LAMMPS_PYTHON.md`。

**力计算方式**：使用 atom forces 模式（`dE/d(pos)`），通过 dummy position 张量做 autograd 叶变量，
梯度存储从 O(npairs) 降至 O(natoms)。Per-atom forces 直接写入 LAMMPS force buffer，
全局 virial 由 LAMMPS 的 `virial_fdotr_compute()` 自动处理。

**模型限制**：仅以下五种模型支持 ML-IAP（因其支持 `precomputed_edge_vec`）：
- `e3nn_layers.py`（spherical）
- `e3nn_layers_channelwise.py`（spherical-save）
- `cue_layers_channelwise.py`（**spherical-save-cue**，cuEquivariance GPU 加速）
- `pure_cartesian_ictd_layers.py`（pure-cartesian-ictd-save）
- `pure_cartesian_ictd_layers_full.py`（pure-cartesian-ictd）

其他模型（如 pure-cartesian、pure-cartesian-sparse 等）暂不支持 ML-IAP 导出。

**spherical-save-cue 的 ML-IAP 说明**：
- 需安装 cuEquivariance：`pip install cuequivariance-torch cuequivariance-ops-torch-cu12`
- 导出时使用 `--tensor-product-mode spherical-save-cue`（或 checkpoint 中已保存该模式）
- 注意：cuEquivariance 内部模块可能无法 pickle，若 `torch.save` 失败，可改用 `pure-cartesian-ictd` 或 `spherical-save` 作为 ML-IAP 替代

### 步骤 1：导出模型

```bash
python -m molecular_force_field.cli.export_mliap your_checkpoint.pth \
    --elements H O \
    --atomic-energy-keys 1 8 \
    --atomic-energy-values -13.6 -75.0 \
    --max-radius 5.0 \
    --output model-mliap.pt
```

### 步骤 2：Python 驱动 LAMMPS

```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

lmp = lammps.lammps()
activate_mliappy(lmp)

model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)

lmp.commands_string("""
units metal
atom_style atomic
read_data your_system.data
pair_style mliap unified model-mliap.pt 0
pair_coeff * * H O
velocity all create 300 12345
fix 1 all nve
thermo 100
run 1000
""")
lmp.close()
```

### 步骤 3：纯 LAMMPS 输入文件

创建 `run.py`：

```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

lmp = lammps.lammps()
activate_mliappy(lmp)
model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)
lmp.file("input.lammps")
lmp.close()
```

`input.lammps` 示例：

```
units metal
atom_style atomic
read_data system.data

pair_style mliap unified model-mliap.pt 0
pair_coeff * * H O

velocity all create 300 12345
fix 1 all nvt temp 300 300 0.1
thermo 100
dump 1 all xyz 100 traj.xyz
run 10000
```

运行（macOS 需设置 `DYLD_LIBRARY_PATH`）：

```bash
export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib"
python run.py
```

**ML-IAP 注意事项：**
- `pair_coeff * * H O` 中元素顺序必须与导出时 `--elements` 一致
- `pair_style mliap unified model.pt 0` 末尾 `0` 表示不包含 ghost 邻居；多层消息传递时改为 `1`
- 使用 `units metal`（eV, Angstrom）

### GPU 加速（ML-IAP-Kokkos）

若 LAMMPS 使用 Kokkos + CUDA 编译，接口会自动检测 GPU 并将 per-atom forces 直接写入 GPU 端的 LAMMPS force buffer（`data.f`），避免 CPU↔GPU 拷贝。全局 virial 由 LAMMPS 的 `virial_fdotr_compute()` 自动计算。

**Python 驱动示例（Kokkos）：**

```python
import torch
import lammps
from lammps.mliap import activate_mliappy_kokkos, load_unified

lmp = lammps.lammps(cmdargs=["-k", "on", "g", "1", "-sf", "kk"])
activate_mliappy_kokkos(lmp)

model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)

lmp.file("input.lammps")
lmp.close()
```

**命令行（多 GPU）：**

```bash
mpirun -np 4 lmp -k on g 4 -sf kk -pk kokkos newton on neigh half -in input.lammps
```

详见 [NVIDIA ML-IAP-Kokkos 教程](https://developer.nvidia.cn/blog/enabling-scalable-ai-driven-molecular-dynamics-simulations/)。

### 步骤 4：Kokkos GPU 加速（参考 MACE/LAMMPS 文档）

若 LAMMPS 已用 Kokkos + CUDA 编译，可通过命令行启用 GPU：

**方式 A：直接运行 LAMMPS 可执行文件**

```bash
lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lammps
```

- `-k on g 1`：启用 Kokkos，使用 1 块 GPU
- `-sf kk`：使用 Kokkos 变体（如 `pair_style mliap/kk`）
- `-pk kokkos newton on neigh half`：Kokkos 包选项

多 GPU 示例：

```bash
mpirun -np 2 lmp -k on g 2 -sf kk -pk kokkos newton on neigh half -in input.lammps
```

**方式 B：Python 驱动 + Kokkos**

若通过 Python 库启动 LAMMPS 并希望使用 Kokkos GPU，需用 `activate_mliappy_kokkos` 替代 `activate_mliappy`：

```python
import torch
import lammps
from lammps.mliap import activate_mliappy_kokkos, load_unified  # Kokkos 版本

lmp = lammps.lammps(cmdargs=["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"])
activate_mliappy_kokkos(lmp)
model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)
lmp.file("input.lammps")
lmp.close()
```

**编译要求**：需 `PKG_KOKKOS=ON`、`Kokkos_ENABLE_CUDA=yes` 等，详见 `molecular_force_field/docs/INSTALL_LAMMPS_PYTHON.md`。
**注意**：Python + Kokkos 联机存在已知问题（如 `mliap_unified_couple_kokkos` 模块不可用），可优先尝试方式 A。

参考：[MACE in LAMMPS with ML-IAP](https://mace-docs.readthedocs.io/en/latest/guide/lammps_mliap.html)

---

## 方式三：fix external / pair_style python

### 1. 生成 LAMMPS 接口文件

使用 `mff-lammps` 命令生成所需的文件：

```bash
mff-lammps path/to/your/model.pth \
    --output-dir ./lammps_setup \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv \
    --device cuda
```

这会生成：
- `lammps_potential.py`: Python 初始化脚本
- `lammps_input.in`: 示例 LAMMPS 输入脚本

### 2. 在 LAMMPS 中使用

#### 方法 A: 使用 `fix python/invoke`（推荐）

```lammps
# 初始化 Python 接口
python lammps_potential.py input

# 设置 Python potential
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "path/to/model.pth" device cuda max_radius 5.0 atomic_energy_file "fitted_E0.csv"

# 设置 pair style
pair_style python 1
pair_coeff * *

# 设置系统（示例）
read_data your_system.data
# 或
create_atoms 1 random 100 12345 NULL

# 运行模拟
run 1000
```

## 单卡运行建议（你当前选择的方式）

- **建议先用单进程/单卡**跑通接口与数值验证：`mpirun -np 1 lmp ...` 或直接 `lmp ...`
- 单卡下不会碰到复杂的 MPI 域分解/ghost 交互问题，定位更快。

## triclinic 盒子支持（重要）

你的模型/邻居构图已经支持 triclinic，但 **LAMMPS 侧需要把 tilt 因子传给 Python**，否则你只能得到正交 cell（会算错）。

本接口支持在 `lammps_potential(...)` 额外传入：
- `xy`, `xz`, `yz`（LAMMPS triclinic tilt factors）

示例（在 LAMMPS 输入脚本里）：

```lammps
variable XY equal xy
variable XZ equal xz
variable YZ equal yz

python lammps_potential.py input

# 注意：下面这一行的 python/invoke 传参格式，取决于你使用的 LAMMPS 版本。
# 关键点是：把 ${XY} ${XZ} ${YZ} 作为额外参数传进 Python 的 lammps_potential(xy=..., xz=..., yz=...)
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "path/to/model.pth" device cuda max_radius 5.0 \
    xy ${XY} xz ${XZ} yz ${YZ}
```

接口内部会按 LAMMPS 常用约定构造 3×3 cell（行向量）：
\[
a=(L_x,0,0),\quad b=(xy,L_y,0),\quad c=(xz,yz,L_z)
\]

如果你用的是不同的 triclinic 约定（比如直接给 h 矩阵），告诉我你 LAMMPS 的版本和调用方式，我可以把“取 box/h 矩阵”的部分改成完全自动。

#### 方法 B: 使用 `pair_style python`（简化版）

```lammps
# 初始化（在 Python 脚本中完成）
python lammps_potential.py input

# 直接使用 pair_style
pair_style python 1 lammps_potential.py lammps_potential_init lammps_potential
pair_coeff * *
```

### 3. 完整示例

#### 示例 1: 简单 MD 模拟

```lammps
# LAMMPS input script for FusedEquiTensorPot

# 初始化 Python 接口
python lammps_potential.py input

# 设置 Python potential
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "combined_model.pth" device cuda max_radius 5.0

# 设置 pair style
pair_style python 1
pair_coeff * *

# 读取系统
read_data system.data

# 设置初始速度
velocity all create 300.0 12345

# 设置积分器
fix 1 all nve

# 输出设置
thermo_style custom step temp pe ke etotal
thermo 100
dump 1 all custom 100 traj.xyz id type x y z fx fy fz

# 运行
run 10000
```

#### 示例 2: NPT 系综（恒温恒压）

```lammps
# 初始化（同上）
python lammps_potential.py input
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "combined_model.pth" device cuda max_radius 5.0
pair_style python 1
pair_coeff * *

# 读取系统
read_data system.data

# NPT 系综
fix 1 all npt temp 300.0 300.0 0.1 iso 1.0 1.0 1.0

# 运行
run 10000
```

## 参数说明

### `lammps_potential_init` 参数

- `checkpoint_path` (必需): 模型检查点文件路径（.pth）
- `device` (可选): 计算设备，`'cuda'` 或 `'cpu'`（默认: `'cuda'`）
- `max_radius` (可选): 邻居搜索的最大半径，单位 Angstrom（默认: 5.0）
- `atomic_energy_file` (可选): 原子基准能量 CSV 文件路径（默认: `fitted_E0.csv`）
- `atomic_energy_keys` (可选): 自定义原子序数列表
- `atomic_energy_values` (可选): 对应的原子能量列表（eV）

### 单位转换

- **能量**: 模型使用 eV，LAMMPS 使用 kcal/mol
  - 转换系数: 1 eV = 23.06035 kcal/mol
- **力**: 模型使用 eV/Å，LAMMPS 使用 kcal/(mol·Å)
  - 转换系数: 1 eV/Å = 23.06035 kcal/(mol·Å)
- **距离**: 两者都使用 Angstrom，无需转换

## 性能优化建议

1. **优先使用 USER-MFFTORCH**: 若需 HPC、无 Python 部署，推荐 `pair_style mff/torch` + Kokkos 纯 C++ 链路
2. **其次 ML-IAP**: 若 LAMMPS 已编译 ML-IAP，推荐 `pair_style mliap unified`，比 fix external 更快
3. **使用 GPU**: fix external 模式下，设置 `device cuda` 可显著加速；ML-IAP 与 USER-MFFTORCH 均支持 Kokkos GPU
4. **邻居列表**: `max_radius` 应与训练时一致
5. **批量处理**: LAMMPS 会自动处理多个时间步，模型在 `eval()` 模式下运行
6. **内存管理**: 大系统可考虑 CPU 模式或减少 `max_num_neighbors`

## 故障排除

### 问题 1: Python 模块导入错误

**错误信息**:
```
ImportError: No module named 'molecular_force_field'
```

**解决方案**:
- 确保已安装 `molecular_force_field` 包：`pip install -e .`
- 或者在 `lammps_potential.py` 中修改 `sys.path` 指向正确的包路径

### 问题 2: CUDA 设备不可用

**错误信息**:
```
RuntimeError: CUDA error: no kernel image is available
```

**解决方案**:
- 使用 CPU 模式：`device cpu`
- 或检查 CUDA 版本兼容性

### 问题 3: 邻居列表计算失败

**错误信息**:
```
RuntimeError: radius_graph failed
```

**解决方案**:
- 检查 `max_radius` 是否合理（通常 3-10 Å）
- 确保系统尺寸足够大（非周期性系统需要足够大的盒子）

### 问题 4: 原子类型不匹配

**错误信息**:
```
KeyError: atomic number not found in energy mapping
```

**解决方案**:
- 确保 `atomic_energy_file` 包含所有出现的原子类型
- 或使用 `atomic_energy_keys` 和 `atomic_energy_values` 手动指定

## 测试接口

**USER-MFFTORCH 接口：**

```bash
bash lammps_user_mfftorch/examples/run_smoke_mfftorch.sh /path/to/lmp cuda
# 或完整 GPU 测试（含 dummy 模型）：
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh --lmp /path/to/lmp --dummy-ictd --elements H O
# spherical-save-cue dummy 测试：
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh --lmp /path/to/lmp --dummy-cue --elements H O --cutoff 5.0 --steps 200
```

**fix external 接口：**

```bash
python -m molecular_force_field.interfaces.lammps_potential \
    path/to/model.pth \
    --device cuda \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv
```

**ML-IAP 接口：**

```bash
python -m molecular_force_field.interfaces.test_mliap
```

会依次测试 atom forces 数值一致性（AtomForcesWrapper）、legacy edge forces 一致性、导出/加载往返、以及 LAMMPS 联机（需已编译 ML-IAP）。

## 高级用法

### 自定义原子能量

如果不想使用 `fitted_E0.csv`，可以直接指定：

```lammps
fix 1 all python/invoke 1 1 1 lammps_potential.py lammps_potential_init lammps_potential \
    "model.pth" device cuda max_radius 5.0 \
    atomic_energy_keys "[1, 6, 7, 8]" \
    atomic_energy_values "[-430.53, -821.03, -1488.19, -2044.35]"
```

### 非周期性系统

对于非周期性系统，接口会自动使用大盒子（100×100×100 Å）来避免计算错误。你仍然需要在 LAMMPS 中正确设置盒子尺寸。

## 注意事项

1. **模型超参数**: 确保 LAMMPS 中使用的 `max_radius` 与训练时一致
2. **原子类型**: LAMMPS 的原子类型（type）应对应模型的原子序数（Z）
   - 更推荐：显式提供 `type_to_Z`（例如 type 1->Si=14），避免静默错误
3. **周期性边界条件**: 接口会自动检测 PBC，但需确保 LAMMPS 中的 PBC 设置正确
4. **性能**: USER-MFFTORCH（纯 C++）> ML-IAP > fix external；HPC 部署优先 USER-MFFTORCH

## 参考

- [USER-MFFTORCH 编译与运行指南](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)（纯 C++ / Kokkos）
- [LAMMPS Python 接口文档](https://docs.lammps.org/Python.html)
- [ML-IAP 接入说明](molecular_force_field/docs/LAMMPS_MLIAP_README.md)
- [LAMMPS 加速方案对比](molecular_force_field/docs/LAMMPS_ACCELERATION.md)
- [FusedEquiTensorPot 主文档](README.md)
- [使用指南](USAGE.md)
