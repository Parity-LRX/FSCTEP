## 目的

这是一个**纯 C++ / LibTorch** 的 smoke test，用来验证你导出的 TorchScript core（`torch.jit.save`）能否在**没有 Python** 的情况下完成：

- TorchScript `forward`
- 用 C++ autograd 计算 \(dE/d(pos)\) 得到 per-atom forces
- 简单的性能基准（ms/iter）

## 1) 先导出可供 C++ 加载的 `core.pt`

注意：`torch.save(LAMMPS_MLIAP_MFF, ...)` 导出的文件**不能**被 C++ `torch::jit::load`。

用仓库内脚本导出真正的 TorchScript `ScriptModule`：

```bash
python molecular_force_field/scripts/export_libtorch_core.py \
  --checkpoint /path/to/ckpt.pth \
  --elements H O \
  --device cuda \
  --out /tmp/core.pt
```

会同时生成 `/tmp/core.pt.json`（元数据）。

## 2) 编译（CMake）

先下载/解压 LibTorch，然后把它的路径传给 CMake。

```bash
cmake -S libtorch_smoketest -B build-libtorch-smoketest \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch"
cmake --build build-libtorch-smoketest -j
```

## 3) 运行

```bash
./build-libtorch-smoketest/libtorch_smoketest --model /tmp/core.pt --device cuda
```

可调参数：

- `--N` 本地原子数（默认 512）
- `--E` 边数（默认 16384）
- `--warmup` 预热次数（默认 5）
- `--iters` 计时迭代次数（默认 20）

