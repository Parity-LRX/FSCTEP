# 常用机器学习力场数据集与 extxyz 转换

这个仓库训练前的原始输入推荐使用 `extxyz`。最低要求是：

- 每帧 comment 行带 `energy=...`
- 原子列至少包含 `species/pos/force`
- 周期体系可额外带 `Lattice=... pbc="T T T"`，以及 `stress=` 或 `virial=`

新增 CLI：

```bash
mff-convert-dataset --help
```

如果你的环境里 `mff-convert-dataset` 触发了仓库顶层 `torch` 导入问题，也可以直接运行独立脚本：

```bash
python scripts/convert_dataset_to_extxyz.py --help
```

## 推荐优先级

### 1. `rMD17`

- 适合：分子势能面、入门验证、快速 smoke test
- 特点：小而标准，单体系训练方便
- 输入格式：官方常见为 `.npz`
- 本仓库转换：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format rmd17 \
  --input rmd17_aspirin.npz \
  --output data/rmd17_aspirin.extxyz
```

### 2. `ANI-1x`

- 适合：有机分子通用势
- 元素：`H/C/N/O`
- 输入格式：常见为 HDF5，每个分子记录下有 `atomic_numbers`、`coordinates`、`wb97x_dz.energy`、`wb97x_dz.forces`
- 本仓库转换：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format ani1x \
  --input ani1x.h5 \
  --output data/ani1x.extxyz
```

如果你手头镜像的键名不同，先看树：

```bash
python scripts/convert_dataset_to_extxyz.py --format ani1x --input ani1x.h5 --inspect
```

然后覆盖键名：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format ani1x \
  --input ani1x.h5 \
  --output data/ani1x.extxyz \
  --energy-key wb97x_dz.energy \
  --force-key wb97x_dz.forces
```

### 3. `QM7-X`

- 适合：小分子高精度多构型数据
- 元素：`C/N/O/S/Cl/H`
- 输入格式：常见为分层 HDF5，单构型组下有 `atNUM`、`atXYZ`、`ePBE0+MBD`、`pbe0FOR`
- 本仓库转换：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format qm7x \
  --input 1000.hdf5 \
  --output data/qm7x_1000.extxyz
```

### 4. `SPICE`

- 适合：更大规模的有机/药化分子训练
- 输入格式：常见为 HDF5，通常是 `atomic_numbers`、`conformations`、`dft_total_energy`、`dft_total_gradient`
- 注意：很多 SPICE 文件存的是 **gradient 而不是 force**，本 CLI 预设会自动取负号
- 本仓库转换：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format spice \
  --input spice.h5 \
  --output data/spice.extxyz
```

## 通用 HDF5 转换

如果你的文件不是上述标准镜像，可以先检查：

```bash
python scripts/convert_dataset_to_extxyz.py --format generic-h5 --input your_data.h5 --inspect
```

再指定键名和单位：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format generic-h5 \
  --input your_data.h5 \
  --output data/custom.extxyz \
  --species-key atomic_numbers \
  --coord-key conformations \
  --energy-key dft_total_energy \
  --force-key dft_total_gradient \
  --distance-unit bohr \
  --energy-unit hartree \
  --force-unit hartree/bohr \
  --force-is-gradient
```

## 转完后接本仓库训练

```bash
mff-preprocess \
  --input-file data/ani1x.extxyz \
  --output-dir data/ani1x_proc \
  --train-ratio 0.95 \
  --max-radius 5.0 \
  --num-workers 8
```

## 抽样转换

大数据集建议先抽样验证：

```bash
python scripts/convert_dataset_to_extxyz.py \
  --format spice \
  --input spice.h5 \
  --output data/spice_10k.extxyz \
  --limit 10000 \
  --stride 10
```
