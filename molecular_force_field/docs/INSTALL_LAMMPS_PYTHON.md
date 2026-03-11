# 本地安装 LAMMPS（Python 接口，CPU / GPU 通用）

本仓库的 LAMMPS 接口通过 `fix python/invoke`、`pair_style python` 或 `pair_style mliap unified` 调用 Python 势函数，因此需要 **带 PYTHON 包且可执行 `lmp` 的 LAMMPS**。支持 CPU 版和 Kokkos GPU 版。

## 方式一：Conda（推荐，CPU）

```bash
conda create -n lammps python=3.10 -y
conda activate lammps
conda install -c conda-forge lammps
```

安装后可直接使用 `lmp` 命令，且通常已包含 PYTHON 包。适合快速验证和 fix external 接口。

## 方式二：从源码编译（CPU）

1. **依赖**：CMake、C++17 编译器（如 GCC/Clang）、Python 3.6+ 及开发头文件。

2. **编译选项**：
   - `BUILD_SHARED_LIBS=on`（必须，供 Python 动态加载）
   - `PKG_PYTHON=on`

3. **示例（在 LAMMPS 源码根目录下）**：

```bash
mkdir build && cd build
cmake ../cmake \
  -D BUILD_SHARED_LIBS=on \
  -D PKG_PYTHON=on \
  -D CMAKE_BUILD_TYPE=Release
make -j
make install
make install-python   # 将 liblammps 与 Python 模块安装到当前 Python 的 site-packages
```

4. 将安装目录下的 `bin` 加入 `PATH`，确认 `lmp -h` 可用。

## 方式三：从源码编译（GPU，Kokkos）

若需使用 ML-IAP 的 Kokkos GPU 加速，需额外启用 `PKG_KOKKOS` 和 `PKG_ML-IAP`。

1. **依赖**：CUDA 11+、CMake、C++17 编译器、Python 3.6+。

2. **编译选项**：
   - `BUILD_SHARED_LIBS=on`
   - `PKG_PYTHON=on`
   - `PKG_KOKKOS=on`
   - `PKG_ML-IAP=on`
   - `MLIAP_ENABLE_PYTHON=on`
   - Kokkos CUDA 相关：`Kokkos_ENABLE_CUDA=yes`、`Kokkos_ARCH_XXX=ON`（按 GPU 架构选择）

3. **示例（NVIDIA GPU）**：

```bash
mkdir build && cd build
cmake ../cmake \
  -D BUILD_SHARED_LIBS=on \
  -D PKG_PYTHON=on \
  -D PKG_KOKKOS=on \
  -D PKG_ML-IAP=on \
  -D MLIAP_ENABLE_PYTHON=on \
  -D Kokkos_ENABLE_CUDA=yes \
  -D Kokkos_ARCH_AMPERE80=ON \
  -D CMAKE_BUILD_TYPE=Release
make -j
make install
make install-python
```

GPU 架构可参考 [LAMMPS Kokkos 文档](https://docs.lammps.org/Speed_kokkos.html)，如 `Kokkos_ARCH_PASCAL60`（P100）、`Kokkos_ARCH_VOLTA70`（V100）、`Kokkos_ARCH_AMPERE80`（A100）等。

## ML-IAP 专用编译选项（CPU 或 GPU）

使用 `pair_style mliap unified` 时，LAMMPS 需编译时开启：

- `PKG_ML-IAP=on`
- `MLIAP_ENABLE_PYTHON=on`
- `PKG_PYTHON=on`
- `BUILD_SHARED_LIBS=on`

若仅用 fix external / pair_style python，则无需 ML-IAP 包。

## 验证

- 终端执行：`lmp -h`，应打印 LAMMPS 帮助。
- **macOS**：若提示找不到 `liblammps.0.dylib`，请设置后再运行 `lmp`：
  ```bash
  export DYLD_LIBRARY_PATH="$HOME/.local/lib:$DYLD_LIBRARY_PATH"
  export PATH="$HOME/.local/bin:$PATH"
  ```
  建议将以上两行加入 `~/.zshrc` 或 `~/.bashrc`。
- Python 中执行：`import lammps; lammps.lammps()`，应能创建 LAMMPS 实例（若已 `make install-python`）。

## 与本仓库接口的测试

- **不依赖 LAMMPS** 的离线自测：  
  `python -m molecular_force_field.test.self_test_lammps_potential`
- **依赖已安装的 LAMMPS** 的联机测试：  
  `python -m molecular_force_field.test.test_lammps_integration`  
  用 LAMMPS Python 库 + fix external + 真实势函数跑 run 0，验证 LAMMPS 能正确调用你的模型计算能量和力。
- **ML-IAP 完整测试**（推荐）：  
  `python molecular_force_field/scripts/test_mliap_full.py`  
  或 `bash molecular_force_field/scripts/test_mliap_full.sh`  
  依次测试：edge forces 一致性、导出/加载、LAMMPS 联机。加 `--kokkos` 可测试 Kokkos GPU。
- **ML-IAP 接口测试**：  
  `python -m molecular_force_field.test.test_mliap`  
  同上，无 Kokkos 选项。
- **Kokkos GPU 安装测试**：  
  `bash molecular_force_field/scripts/test_lammps_kokkos.sh`  
  检查 lmp、Python lammps、CUDA、Kokkos 单步、ML-IAP 模块。
- **ASE vs ML-IAP Kokkos 性能对比**：  
  `python molecular_force_field/scripts/bench_ase_vs_mliap_kokkos.py`  
  多组体系大小，输出 ASE 与 ML-IAP Kokkos 的 ms/step 对比表格。

详见 [LAMMPS_INTERFACE.md](../../LAMMPS_INTERFACE.md) 与 [LAMMPS_MLIAP_README.md](LAMMPS_MLIAP_README.md)。
