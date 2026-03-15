# 使用指南

FusedSCEquiTensorPot 支持**八种等变张量积实现模式**，并可在 `pure-cartesian-ictd`、`pure-cartesian-sparse`、`pure-cartesian-sparse-save` 模式下**嵌入外场**（如电场）和**训练物理张量**（电荷、偶极矩、极化率、四极矩）。包括：
- `spherical`: 基于 e3nn 的球谐函数方法（默认）
- `spherical-save`: channelwise edge conv（e3nn 后端，参数量更少）
- `spherical-save-cue`: channelwise edge conv（cuEquivariance 后端，需可选依赖，GPU 加速）
- `partial-cartesian`: 笛卡尔坐标 + e3nn CG 系数（严格等变，部分使用 e3nn）
- `partial-cartesian-loose`: 近似等变（norm product 近似，部分使用 e3nn）
- `pure-cartesian`: 纯笛卡尔 \(3^L\) 表示（严格等变，速度较慢，完全自实现）
- `pure-cartesian-sparse`: 稀疏纯笛卡尔（严格等变，参数量优化，完全自实现）
- `pure-cartesian-ictd`: ICTD irreps 内部表示（严格等变，速度最快，参数量最少，完全自实现）

所有模式都保持 O(3) 等变性（包括旋转和反射）。详细性能对比见[张量积模式对比](#张量积模式对比)部分。

## 安装

### 1. 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者直接安装包（会自动安装依赖）
pip install -e .
```

### 2. 验证安装

安装完成后，可以通过以下命令验证CLI是否可用：

```bash
mff-preprocess --help
mff-convert-dataset --help
mff-train --help
mff-evaluate --help
mff-export-core --help   # LAMMPS LibTorch 导出
mff-lammps --help        # LAMMPS fix external 接口
mff-init-data --help     # 初始数据集生成（冷启动）
mff-active-learn --help  # 主动学习工作流
python -m molecular_force_field.cli.export_mliap --help  # LAMMPS ML-IAP 导出
```

### 3. LAMMPS 集成

本框架支持三种 LAMMPS 集成方式，详见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)：

| 方式 | 速度 | 要求 | 适用场景 |
|------|------|------|----------|
| **USER-MFFTORCH（LibTorch 纯 C++）** | 最快，无 Python | LAMMPS 编译 KOKKOS + USER-MFFTORCH | HPC、超算、生产部署 |
| **ML-IAP unified** | 较快（约 1.7x fix external） | LAMMPS 编译 ML-IAP | 推荐，支持 GPU |
| **fix external / pair_style python** | 较慢 | 标准 LAMMPS + Python | 快速验证 |

Python API 示例见[示例 5a：LAMMPS LibTorch 接口](#示例-5a-使用-lammps-libtorch-接口usermfftorchhpc-推荐)和[示例 5b：LAMMPS ML-IAP 接口](#示例-5b-使用-lammps-mliap-unified-接口)。

## 完整使用流程

### 步骤 1: 数据预处理

首先，你需要将原始的XYZ文件预处理成库可以使用的格式。

```bash
mff-preprocess \
    --input-file 2000.xyz \
    --output-dir data \
    --train-ratio 0.95 \
    --max-radius 5.0 \
    --num-workers 8
```

**参数说明：**
- `--input-file`: 输入的XYZ文件路径
- `--output-dir`: 输出目录（默认'data'），所有预处理文件会保存在这个目录下
- `--max-atom`: 兼容旧 padded 原始存储的可选参数；默认不需要，`read_{train,val}.h5` 会按样本变长存储
- `--train-ratio`: 训练集比例（默认0.95）
- `--max-radius`: 邻居搜索的最大半径
- `--num-workers`: 并行处理的进程数

**输出文件（在 `data/` 目录下）：**
- `read_train.h5`, `read_val.h5` - 原始原子数据（默认按 `sample_i` 变长存储；传 `--max-atom` 时保留旧 padded 格式）
- `raw_energy_train.h5`, `raw_energy_val.h5` - 原始能量
- `correction_energy_train.h5`, `correction_energy_val.h5` - 校正能量
- `cell_train.h5`, `cell_val.h5` - 晶胞信息
- `stress_train.h5`, `stress_val.h5` - 应力张量（当 XYZ 含 stress/virial 时；否则为零矩阵）
- `processed_train.h5`, `processed_val.h5` - 预处理后的数据（如果使用--preprocess-h5）
- `fitted_E0.csv` - 拟合的原子基准能量（默认：从训练集用最小二乘拟合得到）

**自定义 extxyz 键名覆盖**：

如果你的 `extxyz` 不是默认的 `energy=...` + `Properties=species:...:pos:...:force:...:Z:...`，可以显式指定：

```bash
mff-preprocess \
    --input-file custom.extxyz \
    --output-dir data \
    --energy-key REF_energy \
    --force-key REF_force \
    --species-key elem \
    --coord-key coords \
    --atomic-number-key atomic_number
```

默认搜索规则：
- `energy`: comment 行里的 `energy=...`
- `force`: `force` / `forces` / `f`
- `species`: `species` / `symbol` / `element`
- `coord`: `pos`
- `atomic number`: `Z` / `atomic_number`

### 步骤 2: 训练模型

**方式一：使用已预处理的数据**
```bash
mff-train \
    --data-dir data \
    --train-prefix train \
    --val-prefix val \
    --epochs 1000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 2e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --weight-a-growth 1.05 \
    --weight-b-decay 0.98 \
    --force-shift-value 1.0 \
    --patience 20 \
    --vhat-clamp-interval 2000 \
    --max-vhat-growth 5.0 \
    --max-grad-norm 0.5 \
    --grad-log-interval 500 \
    --dump-frequency 250 \
    --energy-log-frequency 10 \
    --device cuda \
    --dtype float64 \
    --num-workers 8 \
    --max-radius 5.0
```

**应力训练**（可选，用于周期性体系）：
若 XYZ 含 stress/virial 且为 PBC 体系，可启用应力损失：
```bash
mff-train \
    --data-dir data \
    --stress-weight 0.1
```

**自定义原子能量（E0）**（可选）：
- **默认行为**：使用 `--data-dir` 下的 `fitted_E0.csv`（训练集最小二乘拟合得到）
- **方式 1：从 CSV 指定**：

```bash
mff-train \
    --data-dir data \
    --atomic-energy-file data/fitted_E0.csv
```

- **方式 2：直接在 CLI 指定**：

```bash
mff-train \
    --data-dir data \
    --atomic-energy-keys 1 6 7 8 \
    --atomic-energy-values -430.53299511 -821.03326787 -1488.18856918 -2044.3509823
```

**ZBL 短程排斥修正**（可选，推荐用于高能近距离构型）：

ZBL (`Ziegler-Biersack-Littmark`) 是经典的短程核-核排斥势。FSCETP 现在支持把它作为统一短程修正项叠加到模型能量输出上，用来改善极短距离下的物理合理性。典型场景包括：

- 辐照损伤、碰撞级联、离子注入
- 高温或高压下的原子过近接触
- 希望避免 ML 势在未见过的超短程区域出现不物理吸引

训练时开启示例：

```bash
mff-train \
    --data-dir data \
    --tensor-product-mode pure-cartesian-ictd \
    --zbl-enabled \
    --zbl-inner-cutoff 0.6 \
    --zbl-outer-cutoff 1.2 \
    --zbl-exponent 0.23 \
    --zbl-energy-scale 1.0
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--zbl-enabled` | 启用 ZBL 短程排斥修正 |
| `--zbl-inner-cutoff` | 内截止半径（Å）。在此半径内使用完整 ZBL 排斥 |
| `--zbl-outer-cutoff` | 外截止半径（Å）。超过此半径后 ZBL 平滑衰减到 0 |
| `--zbl-exponent` | 屏蔽长度公式中的指数参数，默认常用值为 `0.23` |
| `--zbl-energy-scale` | ZBL 能量整体缩放因子；默认 `1.0` 表示标准强度 |

使用建议：

- 必须满足 `--zbl-inner-cutoff < --zbl-outer-cutoff`
- 若训练集不覆盖极短程构型，建议先从 `0.6 / 1.2 / 0.23 / 1.0` 开始
- 若训练集已经专门包含超短程碰撞区，请谨慎调 cutoff，避免与模型本身学到的短程部分重复计入
- ZBL 以统一修正层方式接入，因此对训练、评估、ASE、active learning、ML-IAP 都适用；但 `core.pt` 导出仍受原本 TorchScript 支持模式限制

常数来源说明：

- 势能中的 `14.399645...` 是 \(\frac{e^2}{4\pi\varepsilon_0}\) 在 `eV·Å` 单位制下的数值
- `0.529177... Å` 是 Bohr 半径 \(a_0\)
- 屏蔽长度中的 `0.8854` 与默认指数 `0.23` 来自标准 ZBL 势的经验参数化：
  \[
  a_{ij} = \frac{0.8854 a_0}{Z_i^{p} + Z_j^{p}}, \quad p=0.23
  \]
- 屏蔽函数
  \[
  \phi(x)=0.1818e^{-3.2x}+0.5099e^{-0.9423x}+0.2802e^{-0.4029x}+0.02817e^{-0.2016x}
  \]
  中的系数与指数也是标准 ZBL 文献的通用拟合参数
- 截断区使用的五次 smoothstep 多项式
  \[
  1-10t^3+15t^4-6t^5
  \]
  不是 ZBL 原始物理常数，而是为了保证内外截止之间能量和力都平滑连续的数值实现

实现上，当前代码采用的 pair 势形式为：

\[
E_{ij}^{\mathrm{ZBL}}(r)
=
s(r)\,\lambda\,
14.3996454784255
\frac{Z_i Z_j}{r}
\phi\!\left(\frac{r}{a_{ij}}\right)
\]

其中 \(\lambda = \texttt{zbl\_energy\_scale}\)，\(s(r)\) 是由 `--zbl-inner-cutoff` 和 `--zbl-outer-cutoff` 定义的平滑截断函数。

checkpoint 与后处理说明：

- 训练后，ZBL 参数会写入 checkpoint metadata
- `mff-evaluate` 默认会自动恢复 checkpoint 中的 ZBL 配置
- `mff-export-core` 与 `python -m molecular_force_field.cli.export_mliap` 也会自动继承 checkpoint 中的 ZBL 配置
- 因此通常只需要在训练时显式开启一次，后续评估与导出无需重复手动填写

`--invariant-channels`（标量不变量通道宽度）：

- 该参数控制若干张量积模式中 product-3 类标量不变量读出块的通道宽度，默认值为 `32`
- 适用模式：`spherical`、`spherical-save`、`partial-cartesian`、`partial-cartesian-loose`、`pure-cartesian`、`pure-cartesian-sparse`、`pure-cartesian-ictd`、`pure-cartesian-ictd-save`
- `spherical-save-cue` 不使用这个参数；即使 checkpoint 里保存了 `invariant_channels`，在 cuEquivariance channelwise 后端中也不会生效
- 之前这些模式的行为等价于固定使用 `32`；现在可以通过 CLI 显式调整，例如：

```bash
mff-train \
    --data-dir data \
    --tensor-product-mode pure-cartesian-ictd \
    --invariant-channels 64
```

- 训练后，`invariant_channels` 会写入 checkpoint 的 `model_hyperparameters`
- 后续恢复训练、`mff-evaluate`、主动学习、thermal loader、`mff-export-core` 与 `python -m molecular_force_field.cli.export_mliap` 都会自动从 checkpoint 恢复该值
- 因此主动学习阶段通常不需要再次手动传入 `--invariant-channels`，除非你明确想覆盖 checkpoint 中保存的结构配置

评估时显式覆盖示例：

```bash
mff-evaluate \
    --checkpoint checkpoint/combined_model_pure_cartesian_ictd.pth \
    --input-file test.xyz \
    --tensor-product-mode pure-cartesian-ictd \
    --zbl-enabled \
    --zbl-inner-cutoff 0.6 \
    --zbl-outer-cutoff 1.2
```

**外场与物理张量训练**（支持 `pure-cartesian-ictd`、`pure-cartesian-sparse`、`pure-cartesian-sparse-save`）：

- **外场嵌入**：将全局张量（如电场，rank=1）注入 conv1，用于场依赖势能
- **物理张量训练**：监督输出电荷、偶极矩、极化率、四极矩（结构级或原子级）

```bash
# 外场（电场）+ 偶极矩/极化率训练
mff-train --data-dir data --tensor-product-mode pure-cartesian-sparse \
  --external-tensor-rank 1 --external-field-file data/efield.npy \
  --physical-tensors dipole,polarizability \
  --dipole-file data/dipole.npy --polarizability-file data/pol.npy \
  --physical-tensor-weights "dipole:2.0,polarizability:1.0"

# 原子级物理张量（需 per-node 标签 HDF5）
mff-train --data-dir data --tensor-product-mode pure-cartesian-sparse-save \
  --extra-per-node-file data/per_atom_labels.h5 \
  --physical-tensors-per-node charge_per_atom,dipole_per_atom
```

**外场与物理张量相关参数：**

| 参数 | 说明 |
|------|------|
| `--external-tensor-rank` | 外场张量秩（如 1=电场），需配合 `--external-field-file` |
| `--external-field-file` | 外场标签文件（.npy/.npz/.h5，形状 B×3） |
| `--charge-file` | 结构级电荷标签（标量/样本） |
| `--dipole-file` | 结构级偶极矩标签（B×3） |
| `--polarizability-file` | 结构级极化率标签（B×3×3） |
| `--quadrupole-file` | 结构级四极矩标签（B×3×3） |
| `--extra-per-node-file` | 原子级标签 HDF5（sample_0, sample_1, ... 含 charge_per_atom 等） |
| `--physical-tensors` | 结构级输出：charge,dipole,polarizability,quadrupole |
| `--physical-tensors-per-node` | 原子级输出：charge_per_atom,dipole_per_atom 等 |
| `--physical-tensor-reduce` | 结构级归约：sum（默认）、mean、none |
| `--physical-tensor-weights` | 损失权重：`charge:1.0,dipole:2.0,...` |
| `--inference-output-physical-tensors` | 保存到 checkpoint：推理时输出物理张量（默认不输出，MD/LAMMPS 仅需能量和力） |

**方式二：自动预处理并训练（一步完成）**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --device cuda
```
如果 `data/` 目录下没有预处理文件，会自动从 `--input-file` 指定的XYZ文件进行预处理。

**快速开始示例（使用默认参数）：**
```bash
# 最简单的训练命令（使用所有默认值）
mff-train --input-file 2000.xyz --device cuda
```

**完整参数示例：**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 2000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 1e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --patience 20 \
    --dump-frequency 250 \
    --device cuda \
    --dtype float64
```

**参数说明：**

**数据参数：**
- `--data-dir`: 数据目录，包含预处理文件（默认'data'）。所有预处理后的数据文件（如processed_train.h5, processed_val.h5）应在此目录下
- `--input-file`: 输入XYZ文件路径，用于自动预处理（可选）。如果指定此参数且数据目录下没有预处理文件，会自动从XYZ文件进行预处理
- `--train-prefix`: 训练数据文件前缀（默认'train'）。用于查找`processed_{train-prefix}.h5`等文件
- `--val-prefix`: 验证数据文件前缀（默认'val'）。用于查找`processed_{val-prefix}.h5`等文件
- `--max-radius`: 邻居搜索的最大半径（默认5.0 Å）。原子间距离超过此值不会被考虑为邻居。此参数在预处理和训练时都需要使用

**基础训练参数：**
- `--epochs`: 训练轮数（默认1000）。训练会运行指定的epoch数，除非早停触发
- `--batch-size`: 批次大小（默认8）。如果GPU内存不足，可以减小此值（如4或2）。注意：验证时batch-size固定为1
- `--checkpoint`: 模型保存路径（默认'combined_model.pth'）。训练完成后会保存最终模型（最佳验证性能的模型）到`checkpoint/`目录下，文件名会包含张量积模式后缀（如`combined_model_pure_cartesian.pth`）。训练过程中也会定期保存中间checkpoint到`checkpoint/`目录
- `--reset-loss-weights`: 从checkpoint恢复训练时，忽略保存的损失权重（a, b），使用命令行参数指定的值（默认False，即使用checkpoint中的权重）
- `--device`: 设备（'cuda'或'cpu'，默认自动检测）。如果有GPU，强烈建议使用'cuda'。分布式训练时会自动为每个进程分配对应的GPU
- `--dtype`: 张量数据类型（'float32'或'float64'，默认'float64'）。选项：'float32'/'float'（32位浮点，速度快但精度略低）或'float64'/'double'（64位浮点，精度高但速度较慢）
- `--seed`: 随机种子（默认42）。用于确保实验可复现性。影响数据shuffle、train/val split等随机操作

**验证和日志参数：**
- `--dump-frequency`: 验证和保存模型的频率（每N个batch，默认250）。每N个batch会进行一次验证并保存模型检查点。注意：验证是基于batch_count触发的，不是固定在每个epoch结束时
- `--train-eval-sample-ratio`: 验证时对训练集采样的比例（0.0-1.0，默认0.2）。设置为1.0表示评估整个训练集，设置为0.2表示只评估20%的训练集（更快）。对于大型数据集，建议使用较小的值以加快验证速度
- `--energy-log-frequency`: 记录能量预测的频率（每N个batch，默认10）。这些日志只写入文件，不输出到控制台
- `--log-val-batch-energy`: 在控制台输出验证阶段每个batch的能量预测（默认False）。如果为False，验证batch能量信息只记录到日志文件，不输出到控制台；如果为True，会同时在控制台和日志文件中输出
- `--save-val-csv`: 保存验证能量和力预测到CSV文件（默认False）。如果启用，验证结果会保存到`validation/`目录下，文件名为`val_energy_epoch{epoch}_batch{batch_count}.csv`和`val_force_epoch{epoch}_batch{batch_count}.csv`
- `--no-save-val-csv`: 禁用保存验证CSV文件（与`--save-val-csv`互斥）。用于减少I/O开销

**早停参数：**
- `--patience`: 早停耐心值（默认20，单位：epoch）。如果连续N个epoch的验证损失（能量损失+力损失+应力损失，未加权）没有改善，训练会自动停止。注意：早停使用的是未加权的验证损失，而不是加权损失（`a × 能量损失 + b × 力损失 + c × 应力损失`）

**学习率参数：**
- `--learning-rate`: 目标学习率（默认1e-3）。这是预热后的学习率，也是训练过程中的主要学习率
- `--min-learning-rate`: 最小学习率（默认2e-5）。学习率不会低于此值，即使经过多次衰减
- `--warmup-batches`: 学习率预热的批次数（默认1000）。前N个batch学习率从`learning_rate × warmup_start_ratio`线性增长到`learning_rate`
- `--warmup-start-ratio`: 预热期间学习率的起始比例（默认0.1）。例如，如果`--learning-rate 1e-3`和`--warmup-start-ratio 0.1`，则学习率会从`1e-4`线性增长到`1e-3`
- `--lr-decay-patience`: 学习率衰减的间隔批次数（默认1000）。每N个batch后，如果验证指标没有改善，学习率会乘以`--lr-decay-factor`
- `--lr-decay-factor`: 学习率衰减因子（默认0.98）。每次衰减时学习率乘以该值（0.98表示减少2%）。学习率调度策略：如果验证损失在`lr-decay-patience`个batch内没有改善，学习率会乘以此因子

**损失权重参数：**
- `--energy-weight` (或 `-a`): 能量损失的初始权重（默认1.0）。总损失 = `a × 能量损失 + b × 受力损失 + c × 应力损失`。训练过程中，`a`会根据`--weight-a-growth`自动增长
- `--force-weight` (或 `-b`): 受力损失的初始权重（默认10.0）。通常受力损失需要更大的权重，因为力的数量远多于能量（每个原子有3个力分量）。训练过程中，`b`会根据`--weight-b-decay`自动衰减
- `--stress-weight` (或 `-c`): 应力损失的权重（默认0.0，即禁用）。设为 > 0 时启用应力训练，通过晶胞应变导数计算应力（σ = (1/V) × dE/dε）。**需要**训练 XYZ 文件中包含 stress 或 virial 数据，且为周期性体系（PBC）。应力单位：eV/Å³
- `--c-min`: 应力权重 `c` 的最小值（默认0.0）
- `--c-max`: 应力权重 `c` 的最大值（默认1000.0）
- `--update-param`: 自动调整权重 `a` 和 `b` 的频率（每N个batch，默认1000）。每N个batch会根据 `--weight-a-growth` 和 `--weight-b-decay` 调整权重。调整公式：`a = a × weight_a_growth`，`b = b × weight_b_decay`
- `--weight-a-growth`: 能量权重 `a` 的增长率（默认1.05，即每次增长5%）。建议值：1.005（慢速，适合超长时间训练）、1.01（中速，更稳定）、1.02（快速）、1.05（超快速）
- `--weight-b-decay`: 受力权重 `b` 的衰减率（默认0.98，即每次减少2%）。建议值：0.995（慢速）、0.99（中速，更稳定）、0.98（快速）
- `--a-min`: 能量权重 `a` 的最小值（默认1.0）。`a`在增长时不会低于此值
- `--a-max`: 能量权重 `a` 的最大值（默认1000.0）。`a`在增长时不会超过此值
- `--b-min`: 受力权重 `b` 的最小值（默认1.0）。`b`在衰减时不会低于此值
- `--b-max`: 受力权重 `b` 的最大值（默认1000.0）。`b`在衰减时不会超过此值
- `--force-shift-value`: 受力标签的缩放系数（默认1.0）。如果力的单位需要转换，可以调整此值。注意：此参数目前未在代码中使用，保留用于未来扩展

**优化器参数：**
- `--vhat-clamp-interval`: 优化器 `v_hat` 钳位的频率（每N个batch，默认2000）。用于防止Adam优化器的二阶矩估计（`v_hat`）过大。每N个batch会检查并限制`v_hat`的增长
- `--max-vhat-growth`: `v_hat` 的最大增长因子（默认5.0）。限制`v_hat`不超过历史最大值的N倍，防止优化器不稳定
- `--max-grad-norm`: 梯度裁剪阈值（默认0.5）。如果梯度范数超过此值，会被裁剪到此值，防止梯度爆炸。使用`torch.nn.utils.clip_grad_norm_`实现
- `--grad-log-interval`: 记录梯度统计信息的频率（每N个batch，默认500）。用于监控训练稳定性。梯度统计信息（范数、最大值、最小值、平均值）会记录到日志文件中

**数据处理参数：**
- `--num-workers`: 数据处理的并行进程数（默认8）。预处理时使用全部进程数，训练时DataLoader会自动分配：训练DataLoader使用`max(1, num_workers // 2)`，验证DataLoader使用`max(1, num_workers // 4)`

**模型架构超参数：**
- `--max-atomvalue`: 原子嵌入的最大原子序数（默认10）。如果数据集包含原子序数 > 10 的元素，需要增大此值。例如，如果数据集包含Cl（原子序数17），需要设置`--max-atomvalue 17`
- `--embedding-dim`: 原子嵌入维度（默认16）。增大此值可以增强模型表达能力，但会增加计算量和显存占用
- `--embed-size`: 读出MLP的隐藏层大小（默认[128, 128, 128]）。可以指定多个值，例如`--embed-size 128 256 128`表示三层隐藏层，大小分别为128、256、128
- `--output-size`: 原子读出MLP的输出大小（默认8）。这是每个原子的特征维度，用于后续的能量和力预测
- `--lmax`: 球谐函数的最高阶数（默认2）。控制不可约表示的最高阶。增大`lmax`可以捕获更高阶的几何信息，但会显著增加计算量。常见值：1（快速，适合简单系统）、2（推荐，平衡性能和精度）、3（高精度，但计算量大）
- `--irreps-output-conv-channels`: irreps_output_conv的通道数（可选，默认None，会使用config中的channel_in，通常为64）。与`--lmax`共同决定irreps形式。例如：
  - `lmax=2, channels=64` → "64x0e + 64x1o + 64x2e"
  - `lmax=1, channels=64` → "64x0e + 64x1o"
  - `lmax=3, channels=64` → "64x0e + 64x1o + 64x2e + 64x3o"
  - 增大通道数可以提升模型容量，但会显著增加显存和计算量
- `--function-type`: 径向基函数类型（默认'gaussian'）。选项：
  - `gaussian`: 高斯基函数（默认，平滑且易于优化，推荐用于大多数场景）
  - `bessel`: 贝塞尔基函数（适合周期性系统）
  - `fourier`: 傅里叶基函数（适合周期性边界条件）
  - `cosine`: 余弦基函数
  - `smooth_finite`: 平滑有限支撑基函数
- `--tensor-product-mode`: 等变张量积实现模式（默认'spherical'）。**本框架支持八种等变张量积模式**，选项：
  - `spherical`: 使用 e3nn 球谐函数张量积（默认，精度高，标准实现，推荐用于大多数场景）
  - `spherical-save`: channelwise edge conv（e3nn 后端，参数量更少）
  - `spherical-save-cue`: channelwise edge conv（cuEquivariance 后端，需 `pip install -e ".[cue]"`，GPU 加速）
  - `partial-cartesian`: 笛卡尔坐标 + e3nn CG 系数（严格等变，参数量减少 17.4%）
  - `partial-cartesian-loose`: 近似等变张量积（norm product 近似，速度较快，参数量减少 17.3%，非严格等变）
  - `pure-cartesian`: 纯笛卡尔张量积（\(3^L\) 表示，严格等变，速度极慢，lmax≥4 时失败，不推荐）
  - `pure-cartesian-sparse`: 稀疏纯笛卡尔张量积（严格等变，参数量减少 29.6%，需设置`--max-rank-other`和`--k-policy`）
  - `pure-cartesian-ictd`: ICTD irreps 内部表示（严格等变，参数量最少减少 72.1%，速度最快，推荐用于大规模训练）
  
  详细性能对比和推荐场景见[张量积模式对比](#张量积模式对比)部分。

**张量积模式特定参数（仅用于pure-cartesian-sparse和pure-cartesian-ictd）：**
- `--max-rank-other`: 稀疏张量积的最大rank（仅用于`pure-cartesian-sparse`模式，默认1）。只允许`min(L1, L2) <= max_rank_other`的相互作用。增大此值允许更多相互作用，但会增加参数量和计算量
- `--k-policy`: 稀疏张量积的K策略（仅用于`pure-cartesian-sparse`模式，默认'k0'）。选项：
  - `k0`: 只保留k=0（促进更高rank，推荐）
  - `k1`: 只保留k=1（收缩到更低rank）
  - `both`: 保留k=0和k=1（更多相互作用，但计算量更大）
- `--ictd-tp-path-policy`: ICTD张量积的路径策略（仅用于`pure-cartesian-ictd`模式，默认'full'）。选项：
  - `full`: 保留所有CG允许的(l1,l2->l3)路径（推荐，性能最好）
  - `max_rank_other`: 只保留`min(l1,l2) <= --ictd-tp-max-rank-other`的路径（减少参数量）
- `--ictd-tp-max-rank-other`: ICTD路径剪枝的最大rank（仅用于`pure-cartesian-ictd`模式，当`--ictd-tp-path-policy=max_rank_other`时使用，默认None）。例如，设置为1时只保留标量/向量耦合

**Long-range 参数（`mff-train` / `mff-evaluate` 共用）：**

下面这一组参数控制 reciprocal / latent long-range 路径。若不显式传入，CLI 会优先从 checkpoint 自动恢复；训练时没有 checkpoint 则使用各自默认值。

| 参数 | 典型值 / 默认 | 说明 |
|------|---------------|------|
| `--long-range-mode` | `none` / `latent-coulomb` / `reciprocal-spectral-v1` / `isolated-far-field-v1` / `isolated-far-field-v2` | 是否启用 long-range 头。`none`=关闭；`latent-coulomb`=实空间 latent-charge 路线，可配 `dense_pairwise` 或 `tree_fmm`；`reciprocal-spectral-v1`=倒空间 long-range 路线，可配 `direct_kspace` 或 `mesh_fft`；`isolated-far-field-v1`=非周期孤立体系的轻量 far-field 校正头，复杂度 `O(N+E)`，不依赖 runtime reciprocal/tree source；`isolated-far-field-v2`=非周期孤立体系的“稀疏显式 far shell + physical kernel + coarse multi-bin tail”校正头，典型复杂度更接近 `O(N log N + E + E_far)`，最坏情形仍可能退化到 `O(N^2)`。 |
| `--long-range-hidden-dim` | `64` | long-range hidden width。`isolated-far-field-v1` 中用于 source/energy head；`isolated-far-field-v2` 中用于 source head、shell encoder、energy head。 |
| `--long-range-boundary` | `nonperiodic` / `periodic` / `slab` | long-range 边界条件。`periodic`=3D 全周期；`slab`=`x/y` 周期、`z` 开边界并配合真空 padding；`nonperiodic` 主要给非周期路径/关闭路径使用。 |
| `--long-range-neutralize` / `--no-long-range-neutralize` | 默认 neutralize | 是否在每个 graph 内先把 latent source 做零均值/中和。通常建议保持开启，能减少整体偏置和 `k≈0` 污染。 |
| `--long-range-filter-hidden-dim` | `64` | reciprocal spectral filter MLP 的隐藏维度；`isolated-far-field-v1` 用于 shell stat MLP；`isolated-far-field-v2` 不使用。 |
| `--long-range-kmax` | `2` | 仅用于 `--long-range-reciprocal-backend direct_kspace`。表示每个方向最多取到的 reciprocal 格点索引。越大越精细，但代价越高。 |
| `--long-range-mesh-size` | `16` | 仅用于 `--long-range-reciprocal-backend mesh_fft`。表示每个晶格方向的均匀 FFT 网格尺寸。越大分辨率越高，代价也越大。 |
| `--long-range-slab-padding-factor` | `2` | 仅用于 `--long-range-boundary slab` 且 `mesh_fft`。表示 slab 法向（当前语义下通常是 `z`）的真空 padding 倍数。 |
| `--long-range-include-k0` / `--no-long-range-include-k0` | 默认不包含 | 是否保留 `k=0` 模式。默认关闭更稳妥；只有你明确要研究该模式贡献时再开启。 |
| `--long-range-source-channels` | `1` | latent reciprocal source 的通道数。可以理解为 long-range source 的“通道宽度”。多数场景先从 `1` 开始。 |
| `--long-range-backend` | `dense_pairwise` / `tree_fmm` | 仅用于 `--long-range-mode latent-coulomb`。`dense_pairwise`=旧的全对全实空间求和，复杂度近似 `O(N^2)`；`tree_fmm`=open-boundary nonperiodic 的 Barnes-Hut 风格树后端，当前支持训练/eval 与 `core.pt -> USER-MFFTORCH` runtime。 |
| `--long-range-reciprocal-backend` | `direct_kspace` / `mesh_fft` | `reciprocal-spectral-v1` 的后端实现。`direct_kspace`=旧版显式倒空间求和；`mesh_fft`=LES 风格 FFT/mesh 路线，主项复杂度更接近 `O(M log M)`，也是当前推荐配置。 |
| `--long-range-energy-partition` | `potential` / `uniform` | reciprocal long-range 总能量如何分回原子。`potential`=按每个原子回插得到的势来分能量，通常更合理，也是推荐值；`uniform`=更均匀地摊回原子。 |
| `--long-range-green-mode` | `poisson` / `learned_poisson` | reciprocal Green kernel 家族。`poisson`=固定泊松核（当前推荐）；`learned_poisson`=在泊松核基础上再加可学习修正。 |
| `--long-range-assignment` | `cic` | 仅用于 `mesh_fft`。粒子与网格之间的上/下采样插值方案。当前实现只开放 `cic`（Cloud-In-Cell），也是推荐选择。 |
| `--long-range-theta` | `0.5` | 仅用于 `--long-range-backend tree_fmm`。opening angle，越小越精确、越慢。 |
| `--long-range-leaf-size` | `32` | 仅用于 `--long-range-backend tree_fmm`。树叶节点最大占据数。 |
| `--long-range-multipole-order` | `0` | 仅用于 `--long-range-backend tree_fmm`。当前 correctness-first 实现只支持 `0`。 |

**推荐组合：**

```bash
--long-range-mode reciprocal-spectral-v1 \
--long-range-reciprocal-backend mesh_fft \
--long-range-green-mode poisson \
--long-range-energy-partition potential \
--long-range-assignment cic
```

**体相 / slab 推荐补充：**

```bash
# 体相
--long-range-boundary periodic \
--long-range-mesh-size 16

# slab
--long-range-boundary slab \
--long-range-mesh-size 16 \
--long-range-slab-padding-factor 2
```

**孤立体系（nonperiodic）推荐：**

```bash
# isolated-far-field-v1：轻量 far-field 校正，O(N+E)，显存友好
--long-range-mode isolated-far-field-v1 \
--long-range-boundary nonperiodic \
--long-range-energy-partition potential

# isolated-far-field-v2：稀疏显式 far shell + physical kernel + coarse multi-bin tail
--long-range-mode isolated-far-field-v2 \
--long-range-boundary nonperiodic \
--long-range-energy-partition potential
```

`v1` 更偏“全局减局部”的轻量统计校正，参数更少；`v2` 当前实现会先构造稀疏显式 far-edge，再做 shell-wise kernel 加权统计，并在显式半径外使用可配置数量的 coarse tail bins 做压缩残差统计。相比 `v1`，`v2` 更适合你希望模型区分不同远场距离带贡献、同时保留显式 physical-kernel bias 的场景。

**isolated-far-field-v1 专用参数**（`mff-train` / `mff-evaluate` / `mff-active-learn` 共用）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--long-range-mode` | — | 必须为 `isolated-far-field-v1` |
| `--long-range-boundary` | `nonperiodic` | 必须为 `nonperiodic`；不支持 periodic/slab |
| `--long-range-hidden-dim` | `64` | source head 与 energy head 的隐藏维度 |
| `--long-range-neutralize` / `--no-long-range-neutralize` | neutralize | 是否对 latent source 做 per-graph 零均值 |
| `--long-range-filter-hidden-dim` | `64` | shell stat MLP 的隐藏维度 |
| `--long-range-source-channels` | `1` | latent source 通道数；`latent_dim = max(source_channels, min(hidden_dim, 16))` |
| `--long-range-energy-partition` | `potential` | `potential` 或 `uniform`；能量如何分回原子 |

**isolated-far-field-v1 不使用的参数**（传入会被忽略或报错）：`--long-range-backend`、`--long-range-reciprocal-backend`、`--long-range-kmax`、`--long-range-mesh-size`、`--long-range-slab-padding-factor`、`--long-range-include-k0`、`--long-range-green-mode`、`--long-range-assignment`、`--long-range-theta`、`--long-range-leaf-size`、`--long-range-multipole-order`。

**isolated-far-field-v2 专用参数**（`mff-train` / `mff-evaluate` / `mff-active-learn` 共用）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--long-range-mode` | — | 必须为 `isolated-far-field-v2` |
| `--long-range-boundary` | `nonperiodic` | 必须为 `nonperiodic`；不支持 periodic/slab |
| `--long-range-hidden-dim` | `64` | `v2` 中 source head、shell encoder、energy head 的隐藏维度；不会直接改变 physical kernel 的形式，但会影响 learned radial gate 和融合头的表达能力。 |
| `--long-range-neutralize` / `--no-long-range-neutralize` | neutralize | 是否对 far latent source 做 per-graph 零均值；推荐保留开启。关闭后更像让模型自己决定全局偏置，但更容易放大低频漂移。 |
| `--long-range-energy-partition` | `potential` | `potential` 或 `uniform`；决定总 long-range 校正如何分回原子。`potential` 通常更合理，也更推荐。 |
| `--long-range-far-source-dim` | `16` | far latent source 维度，即每个原子进入显式 shell / tail 聚合前的 invariant source 宽度。越大表达力越强，但聚合和后续 head 也会更重。 |
| `--long-range-far-num-shells` | `3` | 显式 far shell 个数，必须 `>= 1`。这些 shell 真正基于稀疏 far-edge 按距离分层。显式半径上限不再单独输入，而是由 `far_shell_growth ** far_num_shells` 内部推导。 |
| `--long-range-far-shell-growth` | `2.0` | 壳层几何增长倍数，必须 `> 1`。例如 `2.0` 时，`3` 个显式 shell 语义类似 `(r_c,2r_c]`、`(2r_c,4r_c]`、`(4r_c,8r_c]`，因此显式半径上限自动是 `8 * cutoff`。同时也用于当前 coarse tail bins 的增长尺度。 |
| `--long-range-far-tail` / `--no-long-range-far-tail` | 默认开启 tail | 是否启用 coarse multi-bin tail。关闭后只保留显式 far shells。 |
| `--long-range-far-tail-bins` | `2` | coarse tail bins 个数，必须 `>= 1`。当 tail 开启时，前 `N-1` 个 bins 近似为 `(R_max, growth*R_max]`、`(growth*R_max, growth^2*R_max]` 等几何级数分层，最后一个 bin 为 `(growth^(N-1)*R_max, +inf)`。 |
| `--long-range-far-stats` | `mean,count,mean_r,rms_r` | 当前实现固定支持这一组统计量。这里的 `count` 对显式 shell / tail 实际上对应的是 **kernel-weight sum**，不是纯粹的未加权原子数。 |
| `--long-range-far-source-norm` / `--no-long-range-far-source-norm` | 默认开启 | 是否对 projected far source 做 LayerNorm。开启通常更稳，尤其是在不同体系大小和元素分布差异较大时。 |
| `--long-range-far-gate-init` | `0.0` | shell-encoded far features 的初始标量门控。默认 `0.0` 表示训练初期让新 long-range correction 以更保守的方式介入。 |

`v2` 当前还有一项**内部固定但不可通过 CLI 调整**的实现细节：
- physical kernel 采用 `exp(-kappa r) / r × learned radial gate`，其中 `kappa` 和 gate 参数是模型内部可学习参数，会自动写入 checkpoint / `core.pt`

`v2` 还会**内部推导**显式半径上限：
- `far_max_radius_multiplier = far_shell_growth ** far_num_shells`
- 这样最后一个显式 shell 的上界会与 tail 起点精确对齐，避免出现“空 shell”或“最后一个 shell 与 tail 之间的未编码半径带”

`tail_bins` 的**复杂度与推荐取值**：
- `tail_bins` 增大时，tail 统计循环、`shell_features` 宽度、以及后续 `shell_encoder` / `energy_head` 的输入维度都会近似线性增长，因此它带来的额外开销通常更接近 `O(tail_bins)`，而不是像显式 far-edge 那样直接改变 `E_far`
- 实践上推荐先从 `2` 或 `3` 开始：`2` 适合先做稳定 baseline，`3-4` 适合你怀疑超远距离分层还不够细时再尝试；通常不建议一开始就设得很大，因为更远 bins 的统计本来就更粗，过多 bins 往往收益递减
- 若体系尺寸并不大、显式 shells 已经覆盖了大部分重要远场，`tail_bins=1-2` 通常就够；若体系跨度很大、你明确希望把 `R_max` 之外再细分几层，再考虑增大到 `3-4`

**每个 CLI 的具体行为：**
- `mff-train`：上述 `--long-range-*` 参数会直接决定 `v2` 的结构，并写入 checkpoint metadata。推荐在训练时显式固定这些参数，避免后续恢复/导出时出现结构漂移。
- `mff-evaluate`：同名参数可覆盖 checkpoint 中保存的 `v2` 配置；若不传，则优先恢复 checkpoint 元数据。通常做正式评估时不建议覆盖，除非你明确在做 ablation 或调试。
- `mff-active-learn`：这些参数会原样透传到每一轮重训练的 `mff-train`，用来保证 AL 过程中 `v2` 结构一致。也就是说，AL 不会额外“理解” `v2`，而是负责把训练结构稳定传下去。

**`isolated-far-field-v2` 典型推荐命令：**

```bash
# 1) mff-train：推荐先显式固定 v2 结构参数
mff-train \
  --data-dir data \
  --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode isolated-far-field-v2 \
  --long-range-boundary nonperiodic \
  --long-range-hidden-dim 64 \
  --long-range-neutralize \
  --long-range-energy-partition potential \
  --long-range-far-source-dim 16 \
  --long-range-far-num-shells 3 \
  --long-range-far-shell-growth 2.0 \
  --long-range-far-tail \
  --long-range-far-tail-bins 2 \
  --long-range-far-stats mean,count,mean_r,rms_r \
  --long-range-far-source-norm \
  --long-range-far-gate-init 0.0
```

```bash
# 2) mff-evaluate：通常直接恢复 checkpoint；只有做覆盖实验时才重传结构参数
mff-evaluate \
  --checkpoint model_v2.pth \
  --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode isolated-far-field-v2 \
  --long-range-boundary nonperiodic \
  --long-range-energy-partition potential
```

```bash
# 3) mff-active-learn：把 v2 结构透传到每轮内部 mff-train
mff-active-learn \
  --work-dir al_work \
  --data-dir data \
  --explore-type ase \
  --explore-mode md \
  --label-type pyscf \
  --tensor-product-mode pure-cartesian-ictd \
  --long-range-mode isolated-far-field-v2 \
  --long-range-boundary nonperiodic \
  --long-range-hidden-dim 64 \
  --long-range-energy-partition potential \
  --long-range-far-source-dim 16 \
  --long-range-far-num-shells 3 \
  --long-range-far-shell-growth 2.0 \
  --long-range-far-tail \
  --long-range-far-tail-bins 2 \
  --long-range-far-stats mean,count,mean_r,rms_r \
  --long-range-far-source-norm \
  --long-range-far-gate-init 0.0
```

**isolated-far-field-v2 不使用的参数**（传入会被忽略或报错）：`--long-range-backend`、`--long-range-reciprocal-backend`、`--long-range-kmax`、`--long-range-mesh-size`、`--long-range-slab-padding-factor`、`--long-range-include-k0`、`--long-range-green-mode`、`--long-range-assignment`、`--long-range-theta`、`--long-range-leaf-size`、`--long-range-multipole-order`、`--long-range-source-channels`、`--long-range-filter-hidden-dim`。

**参数组合约束：**

- `--long-range-mode` 当前只支持 `--tensor-product-mode pure-cartesian-ictd` 和 `spherical-save-cue`
- `--long-range-mode latent-coulomb` 当前支持 `boundary=nonperiodic|periodic`；其中 `--long-range-backend tree_fmm` 目前要求 `--long-range-boundary nonperiodic`
- `--long-range-mode isolated-far-field-v1` 仅支持 `--long-range-boundary nonperiodic`，不依赖 `--long-range-backend`；导出 `core.pt` 后由 plain `pair_style mff/torch` 直接执行，无需 runtime reciprocal/tree source
- `--long-range-mode isolated-far-field-v2` 仅支持 `--long-range-boundary nonperiodic`；`--long-range-far-num-shells >= 1`、`--long-range-far-shell-growth > 1`，且显式半径上限内部固定为 `--long-range-far-shell-growth ** --long-range-far-num-shells`；当前 `--long-range-far-stats` 固定为 `mean,count,mean_r,rms_r`
- `isolated-far-field-v2` 的显式部分当前典型复杂度更接近 `O(N log N + E + E_far)`，其中 `E_far` 是显式半径 `cutoff < r <= growth^num_shells * cutoff` 内的稀疏 far-edge 数；若显式半径过大或体系过密，最坏情形仍可能退化到 `O(N^2)`
- `--long-range-reciprocal-backend direct_kspace` 当前要求 `--long-range-boundary periodic`
- `--long-range-reciprocal-backend mesh_fft` 当前支持 `--long-range-boundary periodic` 和 `slab`
- `--long-range-kmax` 主要对应 `direct_kspace`；`--long-range-mesh-size` / `--long-range-assignment` / `--long-range-slab-padding-factor` 主要对应 `mesh_fft`
- `mff-active-learn` 现在也支持透传 `--long-range-*` 相关配置到重训练阶段，避免 AL 轮次中静默丢失新 backend

**这些参数在评估中的作用：**

- 对于 `mff-evaluate`，同名参数可用于**覆盖 checkpoint 中保存的 long-range 结构配置**
- 若不传，评估会优先恢复 checkpoint 中的 long-range 参数
- 因此正常评估/MD/NEB 一般不需要重复手动填写，除非你明确想做结构配置覆盖或调试

**Feature-spectral 参数（`mff-train` / `mff-evaluate` 共用）：**

下面这一组参数控制 feature-space 低秩 FFT 残差块（在 `proj_total` 之前做 `(N,F)->(N,C_lr)->FFT->(N,F)`）。若不显式传入，CLI 会优先从 checkpoint 自动恢复；训练时没有 checkpoint 则使用各自默认值。

| 参数 | 典型值 / 默认 | 说明 |
|------|---------------|------|
| `--feature-spectral-mode` | `none` / `fft` | 是否启用 feature-space 谱残差块。`none`=关闭；`fft`=启用低秩 FFT 残差，在特征空间做 FFT 滤波。 |
| `--feature-spectral-bottleneck-dim` | 由 checkpoint 或默认 | 瓶颈通道数 C_lr。控制从全特征维度压缩到 FFT 网格前的通道数。 |
| `--feature-spectral-mesh-size` | 由 checkpoint 或默认 | 每个晶格方向的均匀 FFT 网格尺寸。越大分辨率越高，代价越大。 |
| `--feature-spectral-filter-hidden-dim` | 由 checkpoint 或默认 | FFT 谱滤波 MLP 的隐藏维度。控制 k 空间滤波器的表达能力。 |
| `--feature-spectral-boundary` | `periodic` / `slab` | 边界模型。`periodic`=3D 全周期；`slab`=x/y 周期、z 真空 padding。 |
| `--feature-spectral-slab-padding-factor` | `2` | 仅用于 `--feature-spectral-boundary slab`。slab 法向（通常 z）的真空 padding 倍数。 |
| `--feature-spectral-neutralize` / `--no-feature-spectral-neutralize` | 默认 neutralize | 是否在 FFT 滤波前对瓶颈特征做 per-graph 零均值。通常建议保持开启。 |
| `--feature-spectral-include-k0` / `--no-feature-spectral-include-k0` | 默认不包含 | 是否保留 k=0 模式。默认关闭更稳妥。 |
| `--feature-spectral-gate-init` | 由 checkpoint 或默认 | 残差门控的初始值。控制 FFT 残差块的初始贡献强度。 |

**推荐组合（feature-spectral）：**

```bash
--feature-spectral-mode fft \
--feature-spectral-boundary periodic \
--feature-spectral-mesh-size 16
```

**slab 推荐补充：**

```bash
--feature-spectral-boundary slab \
--feature-spectral-slab-padding-factor 2
```

**参数组合约束：**

- `--feature-spectral-mode fft` 当前只支持 `--tensor-product-mode pure-cartesian-ictd` 和 `spherical-save-cue`
- `mff-export-core` 会把 `feature_spectral_boundary`、`feature_spectral_slab_padding_factor` 写入 `core.pt.json`，供 USER-MFFTORCH 的 reciprocal solver 使用
- `mff-active-learn` 没有额外的 feature-spectral 专用 CLI；它直接沿用 checkpoint 中保存的配置

**这些参数在评估中的作用：**

- 对于 `mff-evaluate`，同名参数可用于**覆盖 checkpoint 中保存的 feature-spectral 结构配置**
- 若不传，评估会优先恢复 checkpoint 中的 feature-spectral 参数

**SWA 和 EMA 参数：**
- `--swa-start-epoch`: 开始 SWA（Stochastic Weight Averaging）的 epoch（可选，默认None）。启用后，`a` 和 `b` 会在该 epoch 直接切换为 `--swa-a` 和 `--swa-b` 的值，并重置早停计数器（`best_val_loss`重置为inf，`patience_counter`重置为0）。如果未设置，则使用连续的线性增长/衰减策略
- `--swa-a`: SWA 阶段的能量权重 `a`（必须与 `--swa-start-epoch` 一起使用）。当达到`swa-start-epoch`时，`a`会直接切换为此值，不再继续增长
- `--swa-b`: SWA 阶段的力权重 `b`（必须与 `--swa-start-epoch` 一起使用）。当达到`swa-start-epoch`时，`b`会直接切换为此值，不再继续衰减
- `--ema-start-epoch`: 开始 EMA（Exponential Moving Average）的 epoch（可选，默认None）。EMA 模型是主模型参数的指数滑动平均，通常在训练后期启用（建议在总epoch数的60%-80%时启用）。如果未设置，EMA功能被禁用
- `--ema-decay`: EMA 衰减系数（默认 0.999，范围0-1）。越大越平滑，但响应越慢。典型值：0.999（非常平滑，推荐）、0.99（较快响应）
- `--use-ema-for-validation`: 使用 EMA 模型进行验证（而非主模型）。如果启用，验证时会使用EMA权重进行前向传播，通常能获得更稳定的验证结果
- `--save-ema-model`: 在 checkpoint 中保存 EMA 模型权重。如果启用，checkpoint会包含`e3trans_ema_state_dict`，可以从checkpoint恢复EMA模型

**验证加速参数：**
- `--compile-val`: 验证时对 e3trans 使用 `torch.compile`，`none`（默认）或 `e3trans`
- `--compile-val-mode`: 编译模式，如 `reduce-overhead`、`max-autotune`
- `--compile-val-fullgraph`: 强制完整图编译
- `--compile-val-dynamic`: 动态形状

**分布式训练参数（推荐方式：`--n-gpu`）：**
- `--n-gpu`: GPU 数量（默认1）。>1 时自动通过 `torchrun` 启动 DDP，无需手动写 `torchrun` 命令
- `--nnodes`: 节点数（默认1）。>1 时启用多节点 DDP
- `--master-addr`: rendezvous 地址（默认 `auto`，从 SLURM 或 hostname 自动解析）
- `--master-port`: rendezvous 端口（默认 29500）
- `--launcher`: 启动器（`auto` / `local` / `slurm`）

**分布式训练参数（高级 / 手动 torchrun 方式）：**
- `--distributed`: 启用分布式训练（DDP模式）。使用 `--n-gpu` 时自动添加，手动 `torchrun` 时需显式指定
- `--local-rank`: 本地进程rank（默认-1，通常由`torchrun`自动设置，无需手动指定）
- `--backend`: 分布式后端（默认'nccl'，选项：'nccl'或'gloo'）
- `--init-method`: 分布式初始化方法（默认'env://'）

**原子参考能量（E0）参数：**
- `--atomic-energy-file`: 包含原子参考能量的CSV文件路径（可选，默认None）。CSV文件应包含`Atom`和`E0`两列。如果未设置，会使用`{data-dir}/fitted_E0.csv`（由训练集最小二乘拟合得到）
- `--atomic-energy-keys`: 原子序数列表（可选，必须与`--atomic-energy-values`一起使用）。例如：`--atomic-energy-keys 1 6 7 8`表示H、C、N、O
- `--atomic-energy-values`: 对应的原子参考能量值（eV，可选，必须与`--atomic-energy-keys`一起使用）。例如：`--atomic-energy-values -430.53 -821.03 -1488.19 -2044.35`

**训练输出：**

所有输出文件会保存在当前工作目录下：

**模型检查点（保存在`checkpoint/`目录）：**
- `checkpoint/combined_model_{tensor_product_mode}.pth` - 最终模型（最佳验证性能的模型，文件名包含张量积模式后缀，如`_spherical`、`_pure_cartesian`等）
- `checkpoint/combined_model_epoch{epoch}_batch_count{batch_count}_{tensor_product_mode}.pth` - 定期保存的中间模型（频率由`--dump-frequency`控制，默认每250个batch保存一次）

**验证结果（保存在`validation/`目录，仅当启用`--save-val-csv`时）：**
- `validation/val_energy_epoch{epoch}_batch{batch_count}.csv` - 验证集能量预测（包含Target_Energy、Predicted_Energy、Delta三列）
- `validation/val_force_epoch{epoch}_batch{batch_count}.csv` - 验证集力预测（包含Fx_True、Fy_True、Fz_True、Fx_Pred、Fy_Pred、Fz_Pred六列）

**日志和记录文件（保存在当前目录）：**
- `training_YYYYMMDD_HHMMSS.log` - 训练日志（包含详细的batch级别信息，使用RotatingFileHandler，每个文件最大1GB，保留5个备份）
- `loss.csv` - 损失记录（包含每个验证点的训练和验证指标，如 loss、RMSE、MAE 等；启用应力训练时还包含 stress_loss、stress_rmse、stress_mae；启用物理张量训练时还包含 train_phys_loss、val_phys_loss）

### 步骤 3: 评估模型

#### 3.1 静态评估（计算RMSE和MAE）

**注意：评估时需要使用与训练时相同的模型超参数！**

```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --test-prefix test \
    --output-prefix test \
    --batch-size 1 \
    --max-atomvalue 10 \
    --embedding-dim 16 \
    --lmax 2 \
    --irreps-output-conv-channels 64 \
    --function-type gaussian \
    --tensor-product-mode spherical \
    --device cuda
```

**参数说明：**
- `--checkpoint`: 模型检查点路径
- `--test-prefix`: 测试数据文件前缀
- `--output-prefix`: 输出文件前缀
- `--use-h5`: 使用H5Dataset（如果已预处理），否则使用OnTheFlyDataset
- `--batch-size`: 批次大小（默认1）
- `--compile`: 推理加速，`none`（默认）或 `e3trans`（对 e3trans 层使用 `torch.compile`）
- `--compile-mode`: 编译模式，如 `reduce-overhead`、`max-autotune`
- `--max-atomvalue`: 必须与训练时相同
- `--embedding-dim`: 必须与训练时相同
- `--lmax`: 必须与训练时相同
- `--irreps-output-conv-channels`: 必须与训练时相同（如果训练时设置了）
- `--function-type`: 必须与训练时相同
- `--tensor-product-mode`: 必须与训练时相同（支持六种模式：`spherical`、`partial-cartesian`、`partial-cartesian-loose`、`pure-cartesian`、`pure-cartesian-sparse`、`pure-cartesian-ictd`）
- `--output-physical-tensors`: 物理张量输出控制（`auto`=使用 checkpoint 的 inference_output_physical_tensors，`true`=始终输出，`false`=不输出；MD/LAMMPS 仅需能量和力时用 `false`）

**输出文件：**
- `test_loss.csv` - 测试集损失指标（启用物理张量时含 phys_loss）
- `test_energy.csv` - 测试集能量预测
- `test_force.csv` - 测试集力预测

#### 3.2 分子动力学模拟（MD）

使用 ASE 的 Langevin 恒温器进行分子动力学模拟。MD 模拟会自动跳过静态评估，直接进行动力学计算。

**基本用法：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --md-sim \
    --md-input start.xyz \
    --device cuda
```

**完整参数示例：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --md-sim \
    --md-input molecule.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 10000 \
    --md-friction 0.01 \
    --md-relax-fmax 0.05 \
    --md-log-interval 10 \
    --md-output md_traj.xyz \
    --device cuda
```

**MD 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--md-input` | `start_structure.xyz` | 输入结构文件（XYZ 格式） |
| `--md-temperature` | 300.0 | 模拟温度 (K) |
| `--md-timestep` | 1.0 | 时间步长 (fs) |
| `--md-steps` | 10000 | 总步数 |
| `--md-friction` | 0.01 | Langevin 摩擦系数 |
| `--md-relax-fmax` | 0.05 | 预优化力收敛阈值 (eV/Å) |
| `--md-log-interval` | 10 | 日志和轨迹记录间隔 |
| `--md-output` | `md_traj.xyz` | 输出轨迹文件 |
| `--md-no-relax` | False | 跳过初始结构优化 |

**MD 工作流程：**
1. 加载初始结构（`--md-input`）
2. 可选：使用 BFGS 优化器进行结构优化（除非使用 `--md-no-relax`）
3. 初始化 Maxwell-Boltzmann 速度分布
4. 运行 Langevin 动力学模拟
5. 定期保存轨迹和日志

**输出文件：**
- `md_traj.xyz` - MD 轨迹（或 `--md-output` 指定的文件）
- `md_traj_log.txt` - 能量/温度日志（文件名基于 `--md-output`）
- `relaxed_structure.xyz` - 优化后的初始结构（如果进行了预优化）
- `md_relax.log` - 预优化过程的日志

**示例 1：标准 MD 模拟（300K，10 ps）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input molecule.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 10000 \
    --md-output md_300K.xyz \
    --device cuda
```

**示例 2：高温 MD 模拟（500K，50 ps）**
```bash
# 500K 下运行 50 ps
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input reactant.xyz \
    --md-temperature 500 \
    --md-timestep 0.5 \
    --md-steps 100000 \
    --md-friction 0.005 \
    --md-output md_500K.xyz \
    --device cuda
```

**示例 3：跳过预优化直接运行 MD**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input pre_relaxed.xyz \
    --md-no-relax \
    --md-temperature 300 \
    --md-steps 50000 \
    --device cuda
```

**示例 4：长时间 MD 模拟（100 ps）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --md-sim \
    --md-input system.xyz \
    --md-temperature 300 \
    --md-timestep 1.0 \
    --md-steps 100000 \
    --md-log-interval 50 \
    --md-output md_long.xyz \
    --device cuda
```

**注意事项：**
1. **时间步长选择**：通常 0.5-2.0 fs 是安全的。对于轻原子（H），建议使用更小的时间步长（0.5 fs）
2. **摩擦系数**：0.01 是常用值。更小的值（0.001-0.005）适合长时间模拟，更大的值（0.05-0.1）适合快速平衡
3. **预优化**：建议对初始结构进行优化，除非结构已经优化过
4. **周期性边界条件**：如果输入结构包含晶胞信息，会自动使用 PBC
5. **能量单位**：所有能量以 eV 为单位，力以 eV/Å 为单位，应力以 eV/Å³ 为单位

#### 3.3 NEB（Nudged Elastic Band）计算

使用 ASE 的 NEB 方法寻找化学反应的过渡态和能垒。NEB 计算会自动跳过静态评估，直接进行路径优化。

**基本用法：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --device cuda
```

**完整参数示例：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --neb \
    --neb-initial initial.xyz \
    --neb-final final.xyz \
    --neb-images 15 \
    --neb-fmax 0.03 \
    --neb-output neb.traj \
    --device cuda
```

**NEB 参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--neb-initial` | `initial.xyz` | 初始结构（反应物） |
| `--neb-final` | `final.xyz` | 最终结构（产物） |
| `--neb-images` | 10 | 中间图像数量（不包括端点） |
| `--neb-fmax` | 0.05 | 力收敛阈值 (eV/Å)，使用 NEB 投影力 |
| `--neb-output` | `neb.traj` | 输出轨迹文件（ASE Trajectory 格式） |

**NEB 工作流程：**
1. 加载初始和最终结构
2. 创建 NEB 图像链（初始 + N 个中间图像 + 最终）
3. 线性插值生成初始路径
4. 使用 FIRE 优化器优化路径（Climbing Image NEB）
5. 输出优化后的路径和能垒信息

**输出文件：**
- `neb.traj` - NEB 优化轨迹（ASE Trajectory 格式，包含所有图像）

**输出信息包括：**
- 正向能垒 (Forward barrier): E_saddle - E_initial
- 逆向能垒 (Reverse barrier): E_saddle - E_final
- 反应能 (Reaction energy): E_final - E_initial
- 每个优化步骤的详细日志

**准备输入文件示例：**
```bash
# 创建初始结构（反应物）
cat > initial.xyz << EOF
3

O  0.000  0.000  0.000
H  0.960  0.000  0.000
H -0.240  0.930  0.000
EOF

# 创建最终结构（产物）
cat > final.xyz << EOF
3

O  0.000  0.000  0.000
H  1.500  0.000  0.000
H -0.240  0.930  0.000
EOF
```

**示例 1：标准 NEB 计算**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 10 \
    --neb-fmax 0.05 \
    --device cuda
```

**示例 2：高精度 NEB 计算**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 20 \
    --neb-fmax 0.02 \
    --neb-output neb_high_res.traj \
    --dtype float64 \
    --device cuda
```

**示例 3：快速 NEB 计算（较少图像）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --neb \
    --neb-initial reactant.xyz \
    --neb-final product.xyz \
    --neb-images 5 \
    --neb-fmax 0.1 \
    --device cuda
```

**查看和分析 NEB 结果：**
```python
from ase.io import read
import matplotlib.pyplot as plt
import numpy as np

# 读取 NEB 轨迹
images = read('neb.traj', index=':')

# 提取能量
energies = [img.get_potential_energy() for img in images]

# 绘制能量曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(energies)), energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('Energy (eV)', fontsize=12)
plt.title('NEB Energy Profile', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('neb_profile.png', dpi=300)
plt.close()

# 计算能垒和反应能
e_initial = energies[0]
e_saddle = max(energies)
e_final = energies[-1]

forward_barrier = e_saddle - e_initial
reverse_barrier = e_saddle - e_final
reaction_energy = e_final - e_initial

print(f"Initial energy:    {e_initial:.4f} eV")
print(f"Saddle point:      {e_saddle:.4f} eV")
print(f"Final energy:      {e_final:.4f} eV")
print(f"Forward barrier:   {forward_barrier:.4f} eV")
print(f"Reverse barrier:   {reverse_barrier:.4f} eV")
print(f"Reaction energy:   {reaction_energy:.4f} eV")

# 可视化结构（使用 ASE GUI）
from ase.visualize import view
view(images)
```

**使用 ASE 分析 NEB 结果：**
```python
from ase.io import read
from ase.neb import NEBTools

# 读取轨迹
images = read('neb.traj', index=':')

# 使用 NEBTools 分析
nebtools = NEBTools(images)

# 获取能垒
barrier = nebtools.get_barrier()[0]  # 正向能垒
print(f"Forward barrier: {barrier:.4f} eV")

# 获取鞍点索引
saddle_index = nebtools.get_fitted_pes()[1]
print(f"Saddle point at image: {saddle_index}")

# 绘制拟合的势能面
nebtools.plot_band()
```

**注意事项：**
1. **结构要求**：初始和最终结构必须有相同的原子数和原子类型
2. **预优化**：建议先分别优化初始和最终结构，确保它们是局部最小值
3. **图像数量**：更多图像（15-20）提供更高路径分辨率，但计算量更大。10 个图像通常是好的起点
4. **收敛标准**：`--neb-fmax` 使用 NEB 投影力（不是原始原子力），这是正确的收敛判据
5. **Climbing Image**：自动启用 Climbing Image NEB (CI-NEB) 来精确定位鞍点
6. **周期性边界条件**：如果输入结构包含晶胞信息，会自动使用 PBC
7. **插值方法**：使用改进的切线方法（improved tangent）进行路径插值
8. **优化器**：使用 FIRE 优化器，对 NEB 优化特别有效

#### 3.4 声子谱计算（Phonon Spectrum）

计算声子谱（Hessian 矩阵、振动频率）。支持非周期性和周期性体系，默认使用 ASE on-the-fly 邻居构建。

**基本用法：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --device cuda
```

**完整参数示例：**
```bash
mff-evaluate \
    --checkpoint combined_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output my_phonon \
    --max-radius 5.0 \
    --atomic-energy-file fitted_E0.csv \
    --device cuda
```

**声子谱参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--phonon` | False | 启用声子谱计算（必需标志） |
| `--phonon-input` | `structure.xyz` | 输入结构文件（XYZ 格式） |
| `--phonon-relax-fmax` | 0.01 | 结构优化力收敛阈值 (eV/Å) |
| `--phonon-output` | `phonon` | 输出文件前缀（生成 `{prefix}_hessian.npy` 和 `{prefix}_frequencies.txt`） |
| `--phonon-no-relax` | False | 跳过结构优化（如果结构已优化） |
| `--max-radius` | 5.0 | 邻居搜索最大半径 (Å) |
| `--atomic-energy-file` | `fitted_E0.csv` | 原子参考能量文件（CSV 格式，包含 Atom 和 E0 列） |

**声子谱工作流程：**
1. 加载输入结构（`--phonon-input`）
2. 可选：使用 BFGS 优化器进行结构优化（除非使用 `--phonon-no-relax`）
3. 使用 ASE on-the-fly 邻居列表构建图结构（自动处理周期性边界条件）
4. 计算 Hessian 矩阵（能量对位置的二阶导数，形状：`[N_atoms * 3, N_atoms * 3]`）
5. 从 Hessian 矩阵计算声子频率（通过动力学矩阵对角化）
6. 保存结果到文件

**输出文件：**
- `{prefix}_hessian.npy` - Hessian 矩阵（NumPy 格式，单位：eV/Å²）
- `{prefix}_frequencies.txt` - 声子频率列表（单位：cm⁻¹，负值表示虚频/不稳定模式）

**输出文件格式：**

`{prefix}_frequencies.txt` 格式示例：
```
# Phonon frequencies (cm⁻¹)
# Negative values indicate imaginary frequencies (unstable modes)
# Index    Frequency (cm⁻¹)
     0       1234.567890
     1        987.654321
     2        456.789012
   ...
```

**示例 1：非周期性体系（分子）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input water.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output water_phonon \
    --device cuda
```

**示例 2：周期性体系（晶体）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input crystal.xyz \
    --phonon-relax-fmax 0.01 \
    --phonon-output crystal_phonon \
    --max-radius 5.0 \
    --device cuda
```

**示例 3：跳过结构优化（已优化结构）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input pre_relaxed.xyz \
    --phonon-no-relax \
    --phonon-output phonon \
    --device cuda
```

**示例 4：高精度计算（更严格的优化）**
```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --phonon \
    --phonon-input structure.xyz \
    --phonon-relax-fmax 0.005 \
    --phonon-output phonon_high_precision \
    --dtype float64 \
    --device cuda
```

**查看和分析结果：**
```python
import numpy as np
import matplotlib.pyplot as plt

# 读取 Hessian 矩阵
hessian = np.load('phonon_hessian.npy')
print(f"Hessian shape: {hessian.shape}")
print(f"Hessian range: [{hessian.min():.6f}, {hessian.max():.6f}] eV/Å²")

# 检查对称性
is_symmetric = np.allclose(hessian, hessian.T, atol=1e-4)
print(f"Hessian symmetric: {is_symmetric}")

# 读取频率
frequencies = []
with open('phonon_frequencies.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                freq = float(parts[1])
                frequencies.append(freq)

frequencies = np.array(frequencies)
print(f"\nTotal modes: {len(frequencies)}")
print(f"Real modes (≥0): {np.sum(frequencies >= 0)}")
print(f"Imaginary modes (<0): {np.sum(frequencies < 0)}")
print(f"Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}] cm⁻¹")

# 绘制频率分布
plt.figure(figsize=(10, 6))
plt.hist(frequencies[frequencies >= 0], bins=50, alpha=0.7, label='Real frequencies')
if np.any(frequencies < 0):
    plt.hist(frequencies[frequencies < 0], bins=20, alpha=0.7, label='Imaginary frequencies', color='red')
plt.xlabel('Frequency (cm⁻¹)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Phonon Frequency Distribution', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('phonon_frequencies.png', dpi=300)
plt.close()
```

**注意事项：**
1. **结构优化**：计算前建议先优化结构到能量最小值（默认启用）。如果结构未优化，可能出现虚频（负频率）
2. **周期性边界条件**：如果输入结构包含晶胞信息（`Lattice` 属性），会自动使用 PBC 和 on-the-fly 邻居构建
3. **邻居构建**：默认使用 ASE on-the-fly 邻居列表，自动处理非周期性和周期性体系
4. **大系统**：对于 >100 原子的系统，Hessian 计算可能较慢（O(N²) 复杂度）
5. **原子能量**：需要提供正确的原子参考能量（通过 `--atomic-energy-file` 或训练时生成的 `fitted_E0.csv`）
6. **虚频**：如果出现虚频（负值），说明结构不稳定，需要进一步优化或检查结构
7. **Hessian 对称性**：理论上 Hessian 矩阵应该是对称的，数值误差通常 < 1e-4
8. **单位**：
   - Hessian 矩阵：eV/Å²
   - 频率：cm⁻¹
   - 实频：正值（稳定模式）
   - 虚频：负值（不稳定模式）

**依赖要求：**
```bash
pip install scipy  # 声子谱计算需要（用于矩阵对角化）
pip install ase    # 结构处理和邻居列表（通常已安装）
```

## 冷启动：从零生成初始数据集 (mff-init-data)

只有种子结构、没有已标注数据时，`mff-init-data` 一键完成 **扰动 → DFT 标注 → 预处理**：

```bash
mff-init-data --structures water.xyz ethanol.xyz \
    --n-perturb 15 --rattle-std 0.05 \
    --label-type pyscf --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --output-dir data

# 周期性体系
mff-init-data --structures POSCAR.vasp \
    --n-perturb 20 --rattle-std 0.02 --cell-scale-range 0.03 \
    --label-type vasp --vasp-xc PBE --vasp-encut 500 \
    --output-dir data
```

主要参数：`--n-perturb`（扰动数）、`--rattle-std`（位移 σ，Å）、`--cell-scale-range`（晶胞缩放范围，周期性体系）、`--min-dist`（最小原子间距过滤）。详见 [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md)。

---

## 主动学习 (mff-active-learn)

主动学习模块实现 DPGen2 风格的工作流：**训练集成 → 探索（MD/NEB）→ 按力偏差筛选 → DFT 标注 → 合并数据 → 重复**，用于在势能面尚未覆盖的区域自动采点并扩充训练集。支持**单节点**（本机多进程并发标注）和**超算**（SLURM 按结构提交作业）；同一套 DFT 脚本模板可同时用于本地测试（`local-script`）和集群（`slurm`）。

当前主动学习的 ASE 路径也支持 long-range checkpoint：

- 推荐配置：`reciprocal-spectral-v1 + mesh_fft + poisson + potential + cic`
- 支持边界：`periodic` 与 `slab`
- `slab` 语义：`x/y periodic + z vacuum padding`
- 已验证入口：`IdentityLabeler`、`ModelDeviCalculator`、`run_ase_md`、`run_active_learning_loop`

### 基本用法

必选参数：`--explore-type`（`ase` 或 `lammps`）、`--label-type`（见下表）。单阶段模式示例：

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --md-steps 500 --n-iterations 5
```

查看全部参数：

```bash
mff-active-learn --help
```

### Long-range 推荐用法

如果你的 checkpoint 使用 LES 风格 reciprocal long-range，建议训练时采用：

```bash
# periodic
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
    --long-range-mode reciprocal-spectral-v1 \
    --long-range-reciprocal-backend mesh_fft \
    --long-range-boundary periodic \
    --long-range-mesh-size 16 \
    --long-range-green-mode poisson \
    --long-range-energy-partition potential \
    --long-range-assignment cic

# slab
mff-train --data-dir data --tensor-product-mode pure-cartesian-ictd \
    --long-range-mode reciprocal-spectral-v1 \
    --long-range-reciprocal-backend mesh_fft \
    --long-range-boundary slab \
    --long-range-mesh-size 16 \
    --long-range-slab-padding-factor 2 \
    --long-range-green-mode poisson \
    --long-range-energy-partition potential \
    --long-range-assignment cic
```

把这种 checkpoint 直接用于 `mff-active-learn` 即可；主动学习不需要额外的 long-range 专用开关。

对于 `slab` 主动学习，建议初始结构使用带显式 `pbc="T T F"` 的 `extxyz` / `xyz`，避免把 z 方向误当成周期方向。

### 核心参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--work-dir` | `al_work` | 主动学习工作目录 |
| `--data-dir` | `data` | 训练数据目录（通常含 `processed_train.h5`，以及默认变长存储的 `read_train.h5` / `raw_energy_train.h5` 等原始文件） |
| `--init-structure` | 从 data-dir 自动提取 | 一个或多个初始结构路径（多结构并行探索），或包含 .xyz 文件的目录 |
| `--init-checkpoint` | 无 | 可选 warm start checkpoint。第 0 轮可跳过训练直接探索；1 个 checkpoint=bootstrap，`n_models` 个 checkpoint=完整集成 |
| `--n-models` | 4 | 集成模型数量 |
| `--n-iterations` | 20 | 单阶段最大迭代次数（未用 --stages 时） |
| `--explore-type` | 必选 | 探索后端：`ase` 或 `lammps` |
| `--explore-mode` | `md` | 探索方式：`md`（分子动力学）或 `neb`（弹性带） |
| `--explore-n-workers` | `1` | 多结构并行探索线程数；`1`=顺序，`>1`=并发（ThreadPoolExecutor） |
| `--label-type` | 必选 | 标注方式，见下表 |
| `--md-temperature` | 300 | MD 温度 (K) |
| `--md-steps` | 10000 | MD 步数 |
| `--md-timestep` | 1.0 | MD 时间步长 (fs) |
| `--md-friction` | 0.01 | Langevin 摩擦系数 |
| `--md-relax-fmax` | 0.05 | 预优化力收敛阈值 (eV/Å) |
| `--md-log-interval` | 10 | 轨迹记录间隔 |
| `--level-f-lo` / `--level-f-hi` | 0.05 / 0.5 | 力偏差筛选阈值 (eV/Å) |
| `--conv-accuracy` | 0.9 | 收敛判定比例 |
| `--epochs` | 由 mff-train 默认 | 每轮每个模型的训练 epoch 数 |
| `--train-n-gpu` | 1 | 训练每个集成模型使用的 GPU 数（每节点）。1=单进程（兼容 CPU/单卡），>1 自动 torchrun DDP |
| `--train-max-parallel` | 0 (auto) | 同时并行训练的最大模型数。0=自动（可用 GPU ÷ train-n-gpu），1=串行。多节点时强制为1 |
| `--train-nnodes` | 1 | **每个模型**使用的节点数。SLURM 下自动按 `总节点÷nnodes` 并行训练多模型 |
| `--train-master-addr` | auto | rendezvous 地址。`auto`=从 SLURM 或本机 hostname 解析 |
| `--train-master-port` | 29500 | 基础端口。并行模型自动偏移 (`+slot`) 避免端口冲突 |
| `--train-launcher` | auto | 启动器：`auto` / `local` / `slurm`。SLURM 下自动按节点子集分配 `--nodelist` |
| `--resume` | 关闭 | 从 `work_dir/al_state.json` 和已有 `iterations/iter_*` 产物恢复主动学习 |
| `--stages` | 无 | 多阶段 JSON 文件路径 |
| `--device` | `cuda` | 推理设备 |
| `--max-radius` | 5.0 | 邻居搜索最大半径 (Å) |
| `--atomic-energy-file` | `data/fitted_E0.csv` | 原子参考能量 CSV |
| `--neb-initial` / `--neb-final` | 无 | NEB 模式下的初/末结构 |

### 多层筛选

候选构型经过三层筛选后再送标注，显著降低 DFT 成本并提升训练集多样性：

| 层 | 名称 | 说明 |
|----|------|------|
| **Layer 0** | 失败帧恢复 | 可选地将部分 `fail` 帧中最不极端的构型纳入候选 |
| **Layer 1** | 不确定性门控 | 保留 `level_f_lo ≤ max_devi_f < level_f_hi` 的帧 |
| **Layer 2** | 多样性筛选 | 用结构指纹（SOAP / deviation 直方图）+ FPS 选取最大化多样性的子集 |

| 参数 | 默认 | 说明 |
|------|------|------|
| `--diversity-metric` | `soap` | 指纹类型：`soap`（需 dscribe）、`devi_hist`（零额外推理）、`none`（不筛） |
| `--max-candidates-per-iter` | 50 | 多样性筛选后每轮最多保留的候选数 |
| `--soap-rcut` | 5.0 | SOAP 截断半径 (Å) |
| `--soap-nmax` | 8 | SOAP 径向基展开阶数 |
| `--soap-lmax` | 6 | SOAP 角向展开阶数 |
| `--soap-sigma` | 0.5 | SOAP 高斯展宽 |
| `--devi-hist-bins` | 32 | `devi_hist` 直方图桶数 |
| `--fail-strategy` | `discard` | `discard`（丢弃 fail 帧）或 `sample_topk`（取最温和 fail 帧加入候选） |
| `--fail-max-select` | 10 | `sample_topk` 时最多纳入的 fail 帧数 |

> SOAP 指纹需 `dscribe`：`pip install dscribe` 或 `pip install molecular_force_field[al]`。未安装时自动回退为 `devi_hist`。

### 标注类型 (--label-type)

| 类型 | 说明 | 典型用途 |
|------|------|----------|
| `identity` | 用当前 ML 模型预测（不跑 DFT） | 调试、快速测试流程 |
| `pyscf` | PySCF 计算（无需外部二进制） | 小分子、本地验证 |
| `vasp` | VASP（ASE 接口） | 平面波 DFT，单节点或脚本内 MPI |
| `cp2k` | CP2K（ASE 接口） | 高斯+平面波，单节点 |
| `espresso` | Quantum Espresso pw.x（ASE 接口） | 单节点 QE |
| `gaussian` | Gaussian g16/g09（ASE 接口） | 单节点 |
| `orca` | ORCA（ASE 接口） | 单节点 |
| `script` | 用户脚本：`脚本路径 input.xyz output.xyz` | 任意 DFT/程序 |
| `local-script` | 与 SLURM 同格式的脚本模板，**本地执行** | 单节点 + 同一套脚本 |
| `slurm` | 与 local-script 同格式的脚本模板，**每结构提交一个 sbatch 作业** | 超算多节点 |

### CLI 命令示例

**快速测试（不跑 DFT，用 ML 自举）：**

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type identity \
    --md-temperature 300 --md-steps 200 --n-iterations 2 --epochs 5 --n-models 2
```

**PySCF（本地，无需外部 DFT 二进制）：**

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 8 --md-steps 500 --n-iterations 5
```

**从已有 checkpoint 直接开始第 0 轮探索：**

```bash
# 单 checkpoint bootstrap：第 0 轮跳过训练，直接 MD -> 候选/多样性 -> 标注
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --init-checkpoint warm_start.pth \
    --n-models 4 \
    --md-steps 1000 --n-iterations 10

# 若已有完整集成，可直接给 n_models 个 checkpoint
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --init-checkpoint model_0.pth model_1.pth model_2.pth model_3.pth \
    --n-models 4 \
    --md-steps 1000 --n-iterations 10
```

**long-range / slab checkpoint 的主动学习 smoke：**

```bash
mff-active-learn --work-dir al_smoke --data-dir al_smoke_data \
    --explore-type ase --explore-mode md --label-type identity \
    --identity-checkpoint model_0.pth \
    --init-checkpoint model_0.pth model_1.pth \
    --init-structure slab_init.xyz \
    --device cpu \
    --max-radius 5.0 \
    --n-models 2 \
    --n-iterations 1 \
    --md-temperature 10 \
    --md-timestep 0.5 \
    --md-steps 2 \
    --md-friction 0.02 \
    --level-f-lo 0.0 \
    --level-f-hi 0.1 \
    --conv-accuracy 1.1 \
    --no-pre-eval
```

其中 `slab_init.xyz` 建议至少包含：

```xyz
2
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 20.0" pbc="T T F" Properties=species:S:1:pos:R:3
H 1.000000 1.000000 1.000000
H 1.000000 1.000000 18.000000
```

**主动学习中断后恢复：**

```bash
mff-active-learn --work-dir al_work --data-dir data \
    --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --resume
```

**VASP（单节点，ASE 接口）：**

```bash
export ASE_VASP_COMMAND="mpirun -np 4 vasp_std"
export VASP_PP_PATH=/path/to/potpaw_PBE

mff-active-learn --explore-type ase --label-type vasp \
    --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4 \
    --label-n-workers 8 --label-threads-per-worker 4 \
    --md-steps 1000 --n-iterations 10
```

**CP2K：**

```bash
mff-active-learn --explore-type ase --label-type cp2k \
    --cp2k-xc PBE --cp2k-cutoff 600 --md-steps 500 --n-iterations 5
```

**Quantum Espresso：**

```bash
mff-active-learn --explore-type ase --label-type espresso \
    --qe-pseudo-dir /path/to/pseudos \
    --qe-pseudopotentials '{"H":"H.pbe.UPF","O":"O.pbe.UPF"}' \
    --qe-ecutwfc 60 --md-steps 500 --n-iterations 5
```

**Gaussian：**

```bash
mff-active-learn --explore-type ase --label-type gaussian \
    --gaussian-method b3lyp --gaussian-basis 6-31+G* --gaussian-nproc 8 \
    --md-steps 500 --n-iterations 5
```

**ORCA：**

```bash
mff-active-learn --explore-type ase --label-type orca \
    --orca-simpleinput "B3LYP def2-TZVP TightSCF" --orca-nproc 8 \
    --md-steps 500 --n-iterations 5
```

**用户脚本（任意 DFT）：**

```bash
mff-active-learn --explore-type ase --label-type script --label-script ./my_dft.sh \
    --md-steps 500 --n-iterations 5
```

**local-script（本地执行脚本模板）：**

```bash
mff-active-learn --explore-type ase --label-type local-script \
    --local-script-template dft_job.sh \
    --label-n-workers 4 --md-steps 500 --n-iterations 3
```

**SLURM（超算，每结构一个作业）：**

```bash
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template dft_job.sh --slurm-partition cpu \
    --slurm-nodes 1 --slurm-ntasks 32 --slurm-time 04:00:00
```

**多阶段 JSON（如先 300K 再 600K）：**

```bash
mff-active-learn --explore-type ase --label-type pyscf --stages stages.json
```

**NEB 探索：**

```bash
mff-active-learn --explore-type ase --explore-mode neb \
    --neb-initial reactant.xyz --neb-final product.xyz \
    --label-type pyscf --pyscf-method b3lyp --n-iterations 5
```

**多模型并行训练（自动 GPU 分配）：**

```bash
# 8 张 GPU，4 个模型各用 1 张 → 4 个模型同时训练（自动）
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --n-models 4 --train-n-gpu 1 --md-steps 500 --n-iterations 5

# 8 张 GPU，4 个模型各用 2 张 DDP → 4 个模型同时训练（自动 8÷2=4）
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --n-models 4 --train-n-gpu 2 --md-steps 500 --n-iterations 5

# 4 张 GPU，4 个模型各用 4 张 → 串行（自动 4÷4=1）
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --n-models 4 --train-n-gpu 4 --md-steps 500 --n-iterations 5

# 手动指定并行度：8 张 GPU，每模型 1 张，最多同时训 2 个
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --n-models 4 --train-n-gpu 1 --train-max-parallel 2
```

> **GPU 分配逻辑**：系统自动检测可用 GPU 池（尊重 `CUDA_VISIBLE_DEVICES`），按 `available_gpus // train_n_gpu` 计算默认并行度。每个模型通过 `CUDA_VISIBLE_DEVICES` 被分配到不重叠的 GPU 子集。日志输出到 `train_logs/model_*.log`。
>
> - `--train-n-gpu 1`（默认）= 单进程 `python -m`，完全兼容 CPU 和单卡
> - `--train-n-gpu N` (N > 1) = `torchrun --nproc_per_node=N --distributed` DDP
> - `--train-max-parallel 0`（默认）= 自动最大并行；`1` = 串行（原始行为）

**多节点 DDP 训练（超算/集群）：**

```bash
# 场景 1：SLURM 分配 2 节点 × 4 GPU，4 个模型各用 2 节点 → 串行（2÷2=1）
#SBATCH --nodes=2 --gres=gpu:4
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 4 --train-nnodes 2 --n-models 4

# 场景 2：SLURM 分配 8 节点 × 4 GPU，4 个模型各用 2 节点 → 4 模型并行（8÷2=4）
#SBATCH --nodes=8 --gres=gpu:4
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 4 --train-nnodes 2 --n-models 4

# 场景 3：SLURM 分配 2 节点 × 8 GPU，4 模型各用 4 GPU → 4 模型并行（跨节点分发）
# 每节点放 2 个模型（8÷4=2），2 节点 × 2 = 4 个并行 slot
#SBATCH --nodes=2 --gres=gpu:8
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 4 --train-nnodes 1 --n-models 4

# 场景 4：SLURM 分配 4 节点 × 8 GPU，4 模型各用 1 整个节点
#SBATCH --nodes=4 --gres=gpu:8
mff-active-learn --explore-type ase --label-type vasp \
    --train-n-gpu 8 --train-nnodes 1 --n-models 4
```

> **资源分配逻辑（三种模式自动切换）**：
>
> | 条件 | 模式 | 分配策略 |
> |------|------|----------|
> | `nnodes=1`，非 SLURM 或单节点 | **本地** | 本地 GPU 池按 `n_gpu` 切分 |
> | `nnodes=1`，SLURM 多节点 | **跨节点分发** | 每个节点的 GPU 池按 `n_gpu` 切分，模型通过 `srun --nodelist` 分发到不同节点 |
> | `nnodes>1` | **多节点 DDP** | SLURM 节点池按 `nnodes` 分组，每组训练一个模型 |
>
> - 每个并行模型使用独立端口 `master_port + slot`，避免冲突
> - 跨节点分发时，模型通过 `srun --nodes=1 --nodelist=<node>` 在指定节点运行，`CUDA_VISIBLE_DEVICES` 限定 GPU 子集
> - 非 SLURM 环境（`launcher=local`）下多节点默认串行

**DDP 安全保证**：
- 仅 rank 0 写入 checkpoint（`is_main_process` 检查）
- 所有 rank 在 checkpoint 写入后通过 `dist.barrier()` 同步，确保文件完整写入后才继续
- 训练结束后所有 rank 统一调用 `dist.destroy_process_group()` 清理

### 并发与错误处理

| 参数 | 默认 | 说明 |
|------|------|------|
| `--label-n-workers` | 1 | 同时跑多少个结构（进程数） |
| `--label-threads-per-worker` | 1（n_workers>1 时）或自动 | 每个结构内部线程数（如 PySCF/VASP 的 OpenMP） |
| `--label-error-handling` | `raise` | `raise`（任一失败即退出）或 `skip`（跳过失败结构继续） |

建议：`n_workers × threads_per_worker ≤ 总核数`。

### 工作流说明（每轮迭代做什么）

每一轮（iteration）大致执行以下步骤，直到达到该阶段的 `max_iters` 或收敛：

1. **训练集成**：用当前 `data_dir` 中的训练数据训练 `n_models` 个模型（不同随机种子）， checkpoint 写入 `work_dir/checkpoint/`。
2. **探索**：用其中一个模型对初始结构跑 MD（或 NEB），得到轨迹 `work_dir/iterations/iter_<i>/explore_traj.xyz`。
3. **筛选（多层）**：
   - **Layer 0**（失败帧恢复）：若 `--fail-strategy sample_topk`，取最不极端的 fail 帧加入候选。
   - **Layer 1**（不确定性门控）：按 `level_f_lo` / `level_f_hi` 筛选出“不确定”的构型。
   - **Layer 2**（多样性筛选）：用 SOAP / deviation 直方图等结构指纹 + FPS 选取最多 `--max-candidates-per-iter` 个多样化子集。
4. **标注**：对筛选出的构型调用 DFT（或 identity/script），生成带能量、力的 extended XYZ，写入 `iterations/iter_<i>/labeled/`。
5. **合并**：将新标注的数据合并回 `data_dir`（更新 `read_train.h5`、`raw_energy_train.h5`、`processed_train.h5` 等），作为下一轮训练的输入。
   `read_train.h5` 默认按 `sample_i` 变长存储原子数据，不再依赖固定 `max-atom` padding。

使用 `--stages` 时，会按 JSON 中定义的阶段顺序执行；每个阶段内重复上述迭代，阶段之间数据累积。

### 工作目录结构 (--work-dir)

典型布局（默认 `al_work`）：

```
al_work/
├── checkpoint/           # 集成模型 checkpoint（model_0_*.pth, model_1_*.pth, ...）
├── init.xyz              # 若未指定 --init-structure，从此处或 data_dir 提取
└── iterations/
    ├── iter_0/
    │   ├── explore_traj_0.xyz # 多结构模式：第 0 个结构的子轨迹
    │   ├── explore_traj_1.xyz # 多结构模式：第 1 个结构的子轨迹
    │   ├── explore_traj.xyz   # 本轮合并后的探索轨迹
    │   └── labeled/           # 本轮 DFT 标注结果（extended XYZ）
    ├── iter_1/
    │   └── ...
    └── ...
```

### DFT 后端参数摘要

**PySCF**（`--label-type pyscf`）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--pyscf-method` | `b3lyp` | 方法：b3lyp, pbe, hf, mp2 等 |
| `--pyscf-basis` | `sto-3g` | 基组：sto-3g, 6-31g*, def2-svp 等 |
| `--pyscf-charge` | 0 | 总电荷 |
| `--pyscf-spin` | 0 | 2S（未配对电子数） |
| `--pyscf-max-memory` | 4000 | 最大内存 (MB) |
| `--pyscf-conv-tol` | 1e-9 | SCF 收敛阈值 |

**VASP**（`--label-type vasp`）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--vasp-xc` | `PBE` | XC 泛函：PBE, LDA, HSE06 等 |
| `--vasp-encut` | 无 | 平面波截断 (eV) |
| `--vasp-kpts` | `1 1 1` | k 点网格 |
| `--vasp-ediff` | 1e-6 | SCF 收敛阈值 (eV) |
| `--vasp-ismear` | 0 | 展宽类型：0=Gaussian, -5=tetrahedron |
| `--vasp-sigma` | 0.05 | 展宽宽度 (eV) |
| `--vasp-command` | 覆盖 ASE_VASP_COMMAND | 运行命令 |
| `--vasp-cleanup` | False | 成功后删除运行目录 |

**CP2K**：`--cp2k-xc`、`--cp2k-basis-set`、`--cp2k-cutoff`、`--cp2k-max-scf`、`--cp2k-charge`、`--cp2k-command`、`--cp2k-cleanup`。  
**Quantum Espresso**：`--qe-pseudo-dir`（必选）、`--qe-pseudopotentials`（必选，JSON）、`--qe-ecutwfc`、`--qe-kpts`、`--qe-command`、`--qe-cleanup`。  
**Gaussian**：`--gaussian-method`、`--gaussian-basis`、`--gaussian-charge`、`--gaussian-mult`、`--gaussian-nproc`、`--gaussian-mem`、`--gaussian-command`、`--gaussian-cleanup`。  
**ORCA**：`--orca-simpleinput`、`--orca-nproc`、`--orca-charge`、`--orca-mult`、`--orca-command`、`--orca-cleanup`。

### 脚本模板占位符（local-script / slurm）

`--local-script-template` 与 `--slurm-template` 使用相同占位符，在脚本中可写：

| 占位符 | 说明 |
|--------|------|
| `{run_dir}` | 当前结构的运行目录 |
| `{input_xyz}` | 输入 XYZ 路径 |
| `{output_xyz}` | 输出 extended XYZ 路径（脚本需写入此文件） |
| `{job_name}` | 作业/任务名称 |
| `{partition}` | SLURM 分区（仅 slurm） |
| `{nodes}` | 节点数（仅 slurm） |
| `{ntasks}` | 任务数（仅 slurm） |
| `{time}` | 时间限制（仅 slurm） |
| `{mem}` | 内存（仅 slurm） |

脚本需保证：执行结束后 `{output_xyz}` 存在且为 extended XYZ（含 `Properties=species:S:1:pos:R:3:energy:R:1:forces:R:3`，能量 eV，力 eV/Å）。

### SLURM 参数（--label-type slurm）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--slurm-template` | 必选 | 作业脚本模板路径 |
| `--slurm-partition` | `cpu` | 队列/分区名 |
| `--slurm-nodes` | 1 | 每作业节点数 |
| `--slurm-ntasks` | 32 | 每作业任务数 |
| `--slurm-time` | `02:00:00` | 墙钟时间限制 |
| `--slurm-mem` | `64G` | 每作业内存 |
| `--slurm-max-concurrent` | 200 | 队列中最大并发作业数 |
| `--slurm-poll-interval` | 30 | 轮询 squeue 间隔（秒） |
| `--slurm-extra` | 无 | 额外 sbatch 参数，如 `--account=myproject` |
| `--slurm-cleanup` | False | 成功后删除运行目录 |

若某结构的 `output.xyz` 已存在（如上次中断后重跑），会自动跳过该结构的提交（resume）。

### 环境变量

| 变量 | 说明 |
|------|------|
| `ASE_VASP_COMMAND` | VASP 运行命令，如 `mpirun -np 4 vasp_std` |
| `VASP_PP_PATH` | VASP 赝势目录 |
| `ASE_CP2K_COMMAND` | CP2K 运行命令 |
| `CP2K_DATA_DIR` | CP2K 数据目录 |
| `ASE_GAUSSIAN_COMMAND` | Gaussian 命令，如 `g16 < PREFIX.com > PREFIX.log` |

ORCA 和 QE 可通过 `--orca-command`、`--qe-command` 指定，或确保可执行文件在 PATH 中。

### 常见问题（FAQ）

- **初始结构从哪里来？** 未指定 `--init-structure` 时，会从 `--data-dir` 里可用的训练数据提取第一个结构，优先使用 `train.xyz`，否则回退到 `processed_train.h5` / `read_train.h5`。支持传入多个文件或一个目录实现**多结构并行探索**：`--init-structure A.xyz B.xyz` 或 `--init-structure structures/`。
- **如何从已有 checkpoint 直接开始主动学习？** 用 `--init-checkpoint`。若只给 1 个 checkpoint，第 0 轮进入 bootstrap 模式：跳过训练，直接 MD 采样并把探索帧送去多样性筛选/标注；若给 `n_models` 个 checkpoint，则第 0 轮仍跳过训练，但可以直接计算 ensemble deviation。
- **如何从中断处继续？** 重新执行同一条命令并加 `--resume`。程序会读取 `work_dir/al_state.json`，并复用已有的 checkpoint、`explore_traj.xyz`、`model_devi.out`、`candidate.xyz`、`labeled.xyz` 和 `merge.done`，避免重复训练/探索/标注/merge。对标注阶段而言，`slurm` 和 `local-script` 标注器还会进一步按结构级别检查已有的 `output.xyz` 并跳过已完成任务；其他本地标注器不做 `output.xyz` 级 resume，但仍可通过 `--label-error-handling skip` 在单个结构失败后继续剩余任务。
- **输出 XYZ 格式要求？** 标注器输出的 XYZ 需包含 `Properties=species:S:1:pos:R:3:energy:R:1:forces:R:3`，能量 eV，力 eV/Å。
- **训练超参数？** `--epochs` 会传给内部 `mff-train`；当前 CLI 仅暴露 `--epochs`，其余见训练章节。

### 多阶段 JSON (--stages)

使用 `--stages stages.json` 时，命令行中的 `--md-*`、`--n-iterations` 等单阶段参数会被忽略。JSON 格式示例：

```json
[
  {"name": "300K", "temperature": 300, "nsteps": 500, "max_iters": 3,
   "level_f_lo": 0.05, "level_f_hi": 0.5, "conv_accuracy": 0.9},
  {"name": "600K", "temperature": 600, "nsteps": 1000, "max_iters": 3,
   "level_f_lo": 0.05, "level_f_hi": 0.5, "conv_accuracy": 0.9}
]
```

### 更多说明

更完整的示例、多阶段字段说明及进阶用法见 [ACTIVE_LEARNING.md](ACTIVE_LEARNING.md)。

## Python API 使用

### 示例 1: 数据预处理

```python
from molecular_force_field.data.preprocessing import (
    extract_data_blocks,
    fit_baseline_energies,
    compute_correction,
    save_set,
    save_to_h5_parallel
)

# 提取数据块（支持解析 energy / pbc / Lattice / Properties / stress / virial）
all_blocks, all_energy, all_raw_energy, all_cells, all_pbcs, all_stresses = extract_data_blocks('data.xyz')

# 拟合基准能量
keys = np.array([1, 6, 7, 8], dtype=np.int64)
fitted_values = fit_baseline_energies(
    train_blocks, train_raw_E, keys,
    initial_values=np.array([-0.01] * len(keys))
)

# 计算校正能量
train_correction = compute_correction(train_blocks, train_raw_E, keys, fitted_values)

# 保存数据（包含 pbc 和 stress 信息；原始 H5 默认按 sample_i 变长存储）
save_set('train', train_indices, train_blocks, train_raw_E, train_correction, all_cells, pbc_list=all_pbcs, stress_list=all_stresses)

# 预处理H5文件（预计算邻居列表）
save_to_h5_parallel('train', max_radius=5.0, num_workers=8)
```

### 示例 2: 训练模型

```python
import torch
from torch.utils.data import DataLoader
from molecular_force_field.models import E3_TransformerLayer_multi, MainNet
from molecular_force_field.data import H5Dataset
from molecular_force_field.data.collate import collate_fn_h5
from molecular_force_field.training.trainer import Trainer
from molecular_force_field.utils.config import ModelConfig

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_dataset = H5Dataset('train')
val_dataset = H5Dataset('val')

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn_h5,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn_h5,
    num_workers=2,
    pin_memory=True
)

# 模型配置
config = ModelConfig()

# num_interaction 说明：
# - 含义：控制 E3_TransformerLayer_multi 以及各 cartesian/pure-cartesian 变体中的交互(卷积)层数
# - 取值：最小 2，默认 2；n 层时会串联 n 次卷积
# - 影响：n 层会拼接 f1..fn，并将 f_combine_product 输出通道扩展为 (n-1)*32x0e
#
# 用法示例：将交互层数设为 3
#   num_interaction=3

# 初始化模型
model = MainNet(
    input_size=config.input_dim_weight,
    hidden_sizes=config.main_hidden_sizes4,
    output_size=1
).to(device)

e3trans = E3_TransformerLayer_multi(
    max_embed_radius=config.max_radius,
    main_max_radius=config.max_radius_main,
    main_number_of_basis=config.number_of_basis_main,
    irreps_input=config.get_irreps_input_conv_main(),
    irreps_query=config.get_irreps_query_main(),
    irreps_key=config.get_irreps_key_main(),
    irreps_value=config.get_irreps_value_main(),
    irreps_output=config.get_irreps_output_conv_2(),
    irreps_sh=config.get_irreps_sh_transformer(),
    hidden_dim_sh=config.get_hidden_dim_sh(),
    hidden_dim=config.emb_number_main_2,
    channel_in2=config.channel_in2,
    embedding_dim=config.embedding_dim,
    max_atomvalue=config.max_atomvalue,
    output_size=config.output_size,
    embed_size=config.embed_size,
    main_hidden_sizes3=config.main_hidden_sizes3,
    num_layers=config.num_layers,
    num_interaction=2,
    device=device
).to(device)

# 创建训练器
trainer = Trainer(
    model=model,
    e3trans=e3trans,
    train_loader=train_loader,
    val_loader=val_loader,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    device=device,
    config=config,
    learning_rate=2e-4,
    epoch_numbers=1000,
    checkpoint_path='combined_model.pth',
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

# 开始训练
trainer.run_training()
```

### 示例 3: 评估模型

```python
import torch
from torch.utils.data import DataLoader
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.data import OnTheFlyDataset
from molecular_force_field.data.collate import on_the_fly_collate
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.utils.config import ModelConfig

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 加载测试数据集
test_dataset = OnTheFlyDataset(
    'read_test.h5',
    'raw_energy_test.h5',
    'cell_test.h5',
    max_radius=5.0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=on_the_fly_collate,
    num_workers=10
)

# 评估
evaluator = Evaluator(
    model=e3trans,
    dataset=test_dataset,
    device=device,
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

metrics = evaluator.evaluate(test_loader, output_prefix='test')
print(f"Energy RMSE: {metrics['energy_rmse']:.6f}")
print(f"Force RMSE: {metrics['force_rmse']:.6f}")
```

### 示例 4: 使用 ASE Calculator 进行 MD 模拟

```python
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase import units
from ase.md import MDLogger
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(
    max_embed_radius=config.max_radius,
    main_max_radius=config.max_radius_main,
    main_number_of_basis=config.number_of_basis_main,
    irreps_input=config.get_irreps_output_conv(),
    irreps_query=config.get_irreps_query_main(),
    irreps_key=config.get_irreps_key_main(),
    irreps_value=config.get_irreps_value_main(),
    irreps_output=config.get_irreps_output_conv_2(),
    irreps_sh=config.get_irreps_sh_transformer(),
    hidden_dim_sh=config.get_hidden_dim_sh(),
    hidden_dim=config.emb_number_main_2,
    channel_in2=config.channel_in2,
    embedding_dim=config.embedding_dim,
    max_atomvalue=config.max_atomvalue,
    output_size=config.output_size,
    embed_size=config.embed_size,
    main_hidden_sizes3=config.main_hidden_sizes3,
    num_layers=config.num_layers,
    num_interaction=2,
    function_type_main=config.function_type,
    device=device
).to(device)

e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 创建计算器
atomic_energies_dict = {
    1: -430.53299511,
    6: -821.03326787,
    7: -1488.18856918,
    8: -2044.3509823
}
calc = MyE3NNCalculator(e3trans, atomic_energies_dict, device, max_radius=5.0)

# 读取结构
atoms = read('structure.xyz')
atoms.set_calculator(calc)

# 可选：预优化结构
print("Relaxing structure...")
opt = BFGS(atoms, logfile='relax.log')
opt.run(fmax=0.05)
write('relaxed_structure.xyz', atoms)

# 初始化速度分布
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

# 设置 MD 参数
timestep = 1.0  # fs
temperature = 300.0  # K
friction = 0.01
n_steps = 10000
log_interval = 10

# 创建 Langevin 动力学
dyn = Langevin(
    atoms, 
    timestep * units.fs, 
    temperature_K=temperature, 
    friction=friction
)

# 添加日志和轨迹记录
dyn.attach(MDLogger(dyn, atoms, 'md_log.txt', header=True, mode="w"), interval=log_interval)
dyn.attach(lambda: write('md_traj.xyz', atoms, append=True), interval=log_interval)

# 运行 MD
print(f"Starting MD: {n_steps} steps ({n_steps * timestep / 1000:.2f} ps)")
dyn.run(n_steps)
print("MD simulation completed!")
```

### 示例 5: 使用 ASE NEB 计算反应能垒

```python
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import FIRE
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch

# 加载模型（同上）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 创建计算器
atomic_energies_dict = {
    1: -430.53299511,
    6: -821.03326787,
    7: -1488.18856918,
    8: -2044.3509823
}
calc = MyE3NNCalculator(e3trans, atomic_energies_dict, device, max_radius=5.0)

# 读取初始和最终结构
initial = read('reactant.xyz')
final = read('product.xyz')

# 可选：预优化端点
print("Optimizing initial structure...")
initial.set_calculator(calc)
opt_initial = BFGS(initial)
opt_initial.run(fmax=0.05)

print("Optimizing final structure...")
final.set_calculator(calc)
opt_final = BFGS(final)
opt_final.run(fmax=0.05)

# 创建 NEB 图像
n_images = 10  # 中间图像数量
images = [initial]
images += [initial.copy() for _ in range(n_images)]
images += [final]

# 设置计算器和晶胞
for image in images:
    if initial.cell.any():
        image.set_cell(initial.cell)
        image.set_pbc(initial.pbc)
    else:
        image.set_cell([100, 100, 100])
        image.set_pbc(False)
    image.set_calculator(calc)

# 创建 NEB 对象
neb = NEB(images, climb=True, method='improvedtangent', allow_shared_calculator=True)
neb.interpolate()

# 优化路径
optimizer = FIRE(neb, trajectory='neb.traj')

# 定义日志函数
def log_neb_status():
    energies = [img.get_potential_energy() for img in images]
    neb_forces = neb.get_forces()
    n_images_total = len(images)
    n_atoms = len(images[0])
    forces_reshaped = neb_forces.reshape(n_images_total, n_atoms, 3)
    intermediate_forces = forces_reshaped[1:-1]
    max_force = (intermediate_forces**2).sum(axis=2).max()**0.5
    print(
        f"NEB step {optimizer.nsteps:4d} | "
        f"E_min = {min(energies):.6f} eV | "
        f"E_max = {max(energies):.6f} eV | "
        f"Barrier = {max(energies) - energies[0]:.4f} eV | "
        f"F_max = {max_force:.4f} eV/Å"
    )

optimizer.attach(log_neb_status, interval=1)

# 运行优化
print("Starting NEB optimization...")
optimizer.run(fmax=0.05)

# 输出最终结果
energies = [img.get_potential_energy() for img in images]
e_initial = energies[0]
e_saddle = max(energies)
e_final = energies[-1]

print("\n" + "=" * 60)
print("NEB Results:")
print(f"  Initial energy:    {e_initial:.4f} eV")
print(f"  Saddle point:      {e_saddle:.4f} eV")
print(f"  Final energy:      {e_final:.4f} eV")
print(f"  Forward barrier:   {e_saddle - e_initial:.4f} eV")
print(f"  Reverse barrier:   {e_saddle - e_final:.4f} eV")
print(f"  Reaction energy:   {e_final - e_initial:.4f} eV")
print("=" * 60)
```

### 示例 5a: 使用 LAMMPS LibTorch 接口（USER-MFFTORCH，HPC 推荐）

**USER-MFFTORCH** 是自定义 LAMMPS 包，提供 `pair_style mff/torch`，在 C++ 侧用 LibTorch 加载 TorchScript 模型，**运行时完全不需要 Python**，适合超算与生产部署。

**模型限制**：目前支持 `pure-cartesian-ictd` 系列和 `spherical-save-cue` 模型。元素顺序、cutoff 需与导出时一致。

**步骤 1：导出 core.pt**（需 Python，一次性）：
```bash
mff-export-core \
  --checkpoint model.pth \
  --elements H O \
  --device cuda \
  --dtype float32 \
  --e0-csv fitted_E0.csv \
  --out core.pt
```

若 checkpoint 训练时已启用 ZBL，则导出的 `core.pt` 会自动包含相同的 ZBL 短程排斥修正；导出命令本身不需要额外再写 `--zbl-*` 参数。

**checkpoint 自动恢复说明**：

- `mff-export-core` 会优先从 checkpoint 自动恢复模型结构超参数，例如 `tensor_product_mode`、`max_radius`、`num_interaction`
- `mff-export-core` 现在**默认嵌入 E0**；只有显式传 `--no-embed-e0` 时才导出纯网络能量
- 新 checkpoint 还会保存 `atomic_energy_keys/atomic_energy_values`，因此多数情况下可直接使用 checkpoint 中的 E0；若显式传 `--e0-csv`，则以 `--e0-csv` 为准
- 如果你显式传了同名 CLI 参数，则**以 CLI 为准**
- 因此日常推荐做法是：只在你明确想覆盖 checkpoint 配置时，才手动传这些结构参数
- 老 checkpoint 若未保存 E0，仍会按旧逻辑回退到本地 `fitted_E0.csv`
- 这不会改变 checkpoint 内容本身，只是改变“导出时如何解析参数”的优先级

**与 long-range 相关的导出参数：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `--export-reciprocal-source` | 关闭 | 让导出的 `core.pt` 额外输出最后一个 long-range source tensor 槽位（历史名字仍为 `reciprocal_source`）。对于 `reciprocal-spectral-v1` / feature-spectral FFT，这会供 USER-MFFTORCH 的 mesh/FFT runtime 使用；对于 `latent-coulomb + tree_fmm`，这会供 open-boundary tree runtime 使用。`isolated-far-field-v1` / `isolated-far-field-v2` 不需要此选项，其 long-range 逻辑已内嵌在 core 前向中。 |

补充说明：

- `mff-export-core` 不需要你再次手动传入 `--long-range-*` 或 `--feature-spectral-*` 这一整组参数；这些会从 checkpoint 元数据自动写入 `core.pt.json`
- 运行时会继续读取的 long-range 元数据包括：`long_range_runtime_backend`、`long_range_source_kind`、`long_range_source_channels`、`long_range_boundary`、`long_range_energy_partition`、`long_range_green_mode`
- 若导出的是 `latent-coulomb + tree_fmm`，还会写入 `long_range_theta`、`long_range_leaf_size`、`long_range_multipole_order`、`long_range_screening`、`long_range_softening`、`long_range_energy_scale`
- 运行时会继续读取的 feature-spectral 元数据包括：`feature_spectral_boundary`、`feature_spectral_slab_padding_factor`
- 当前推荐的 long-range 导出前训练配置：`reciprocal-spectral-v1 + mesh_fft + poisson + potential + cic`
- `isolated-far-field-v1` / `isolated-far-field-v2` 导出后 `long_range_runtime_backend` 都为 `none`，由 `pair_style mff/torch` 直接执行；其中 `v2` 当前实现包含稀疏显式 far shells、内部可学习 screened-Coulomb × learned-radial-gate kernel，以及可配置 bin 数的 coarse tail
- `tree_fmm` 当前 USER-MFFTORCH runtime 支持 `boundary=nonperiodic`、`latent_charge` 标量 source、`multipole_order=0`
- 多 rank / 多 GPU 下的 `tree_fmm` 现已支持 `np=1` 与 `np>1` 一致性 smoke；在 CUDA 运行时，local tree build / local leaf interaction / remote summary evaluation / imported near-field evaluation 现在默认优先走 CUDA-first 线性树主路径
- 运行时环境变量现在建议按下面理解：
  `MFF_TREE_FMM_GPU_AWARE_MPI=1` 表示“请求” device-pointer MPI collective；
  `MFF_TREE_FMM_ASSUME_GPU_AWARE_MPI=1` 表示跳过保守探测、强制启用；
  `MFF_TREE_FMM_DEVICE_LOCAL_EVAL=1` 保留为旧名字兼容别名；
  `MFF_TREE_FMM_REUSE_POSITION_TOL=<tol>` 表示在相邻 MD 步位移较小时复用缓存线性树拓扑

**步骤 2：编译 LAMMPS**：启用 `PKG_KOKKOS`、`PKG_USER-MFFTORCH`，详见 [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)。

**步骤 3：运行 LAMMPS**（纯 LAMMPS，无 Python）：
```bash
# 设置 LibTorch 动态库路径
export LD_LIBRARY_PATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH"

# 使用 Kokkos GPU 运行
lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch
```

**LAMMPS 输入示例**（`in.mfftorch`）：
```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin
pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

velocity all create 300 42
fix 1 all nve
thermo 20
run 200
```

补充说明：

- `pair_style mff/torch` 本身没有单独的 `zbl` 关键字
- 只要 `core.pt` 是从启用 ZBL 的 checkpoint 导出的，短程排斥已经内嵌在 TorchScript 模型里
- 因此运行时只需保证 `core.pt` 路径、元素顺序和导出时一致即可

**USER-MFFTORCH 注意事项**：

- 当前 `mfftorch` 已验证并支持 4 种常见组合：
  1. 无外场、无物理张量：只输出能量和力
  2. 有外场、无物理张量：输出能量和力，并接收运行时 `field`/`field6`/`field9`
  3. 无外场、有物理张量：输出能量、力，以及 `compute mff/torch/phys` 暴露的 tensor
  4. 有外场、有物理张量：同时支持外场输入和 tensor 输出
- 运行时外场支持：
  - `field v_Ex v_Ey v_Ez`：rank-1 外场，适合 `l=1` 向量情形
  - `field6` / `field9`：rank-2 外场输入
- 当前 LAMMPS 端固定暴露的物理量为：
  - 结构级：`charge`、`dipole`、`polarizability`、`quadrupole`
  - 原子级：`charge_per_atom`、`dipole_per_atom`、`polarizability_per_atom`、`quadrupole_per_atom`
- 缺失的物理量是允许的：若某个 head 没训练，对应 `mask=0`，输出自动填 0
- 当前固定 schema 与 `l` 的典型对应关系：
  - `charge` / `charge_per_atom`：`l=0`
  - `dipole` / `dipole_per_atom`：`l=1`
  - `polarizability`：通常是 rank-2 笛卡尔张量，对应 `l=0+2`
  - `quadrupole`：通常对应 `l=2`
- 自定义 physical head 名称当前不会自动暴露到 `compute mff/torch/phys`
- 当前导出到 LAMMPS 的 physical heads 默认假定 `channels_out == 1`

**推荐测试命令**（GPU 脚本）：

```bash
# 1) 无外场、无物理张量：只测 energy/force
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --elements H O --cutoff 5.0 --steps 50

# 2) 有外场、无物理张量
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --elements H O \
  --field-values 0.0 0.0 0.01 --cutoff 5.0 --steps 50

# 3) 无外场、有物理张量
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --dummy-phys-heads --test-phys-compute \
  --elements H O --cutoff 5.0 --steps 50

# 4) 有外场、有物理张量
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp --dummy-ictd --dummy-phys-heads --test-phys-compute \
  --elements H O --field-values 0.0 0.0 0.01 --cutoff 5.0 --steps 50
```

**双卡 / 多卡备注**：

- 若你的机器是“一张 GPU 对应一个 MPI rank”，推荐先用脚本生成 `core.pt` 和 `in.corept`，再手动用你自己的 MPI 启动方式运行
- 常见启动方式如下：

```bash
mpirun -np 2 --allow-run-as-root --bind-to none \
  bash -c 'export CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK; \
  /path/to/lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in /tmp/mff_test/in.corept'
```

**Feature-space FFT 训练 smoke**：

```bash
# 单卡 dry run：ICTD + CUE
bash molecular_force_field/test/run_feature_fft_train_smoketest.sh \
  --mode both \
  --device cuda \
  --dtype float32
```

```bash
# 多卡 dry run：2 GPU DDP
bash molecular_force_field/test/run_feature_fft_train_smoketest.sh \
  --mode both \
  --device cuda \
  --dtype float32 \
  --n-gpu 2
```

这个脚本会：
- 自动生成非退化的 periodic toy 数据
- 先预处理成 `processed_train.h5 / processed_val.h5`
- 再运行带 `feature_spectral_mode=fft` 的 `mff-train`
- 支持 `--mode ictd|cue|both`

**Feature-space FFT LAMMPS smoke**：

```bash
# 单卡 orthogonal smoke
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp \
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
  --lmp /path/to/lmp \
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
  --lmp /path/to/lmp \
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
  --lmp /path/to/lmp \
  --dummy-ictd \
  --test-feature-spectral-fft-slab \
  --elements H \
  --dtype float32 \
  --cutoff 5.0
```

```bash
# 2D slab z-open sanity：验证 z 方向没有被误当成周期方向
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp \
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
  --lmp /path/to/lmp \
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
  --lmp /path/to/lmp \
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

**isolated-far-field-v1 LAMMPS smoke（open-boundary nonperiodic）：**

```bash
# 单卡 smoke：LAMMPS vs Python eager 一致性
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp \
  --dummy-ictd \
  --test-isolated-far-field \
  --elements H \
  --dtype float32 \
  --cutoff 5.0
```

```bash
# 吞吐量对比：baseline vs isolated-far-field-v1（open-boundary MD）
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp \
  --dummy-ictd \
  --test-isolated-far-field \
  --compare-throughput \
  --elements H O \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200
```

**isolated-far-field-v2 LAMMPS smoke（open-boundary nonperiodic）：**

```bash
# 单卡 smoke：LAMMPS vs Python eager 一致性
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp \
  --dummy-ictd \
  --test-isolated-far-field-v2 \
  --elements H \
  --dtype float32 \
  --cutoff 5.0
```

```bash
# 吞吐量对比：baseline vs isolated-far-field-v2（open-boundary MD）
bash molecular_force_field/test/run_gpu_lammps_with_corept.sh \
  --lmp /path/to/lmp \
  --dummy-ictd \
  --test-isolated-far-field-v2 \
  --compare-throughput \
  --elements H O \
  --dtype float32 \
  --cutoff 5.0 \
  --steps 200
```

说明：
- 当前 reciprocal solver 已经是“distributed-contract + correctness-first”的首版：去掉了原子全量 gather，默认走 mesh-reduce / slab-distributed FFT 路径
- 当前推荐模式是 `--gpu 1 --np N`，即每个 MPI rank 绑定一张 GPU
- 多 MPI rank 下仍可能因为非 GPU-aware MPI 而发生通信阶段的 host staging；这不是最终的全程 GPU-resident 版本
- 当前版本已经把 `phi/gx/gy/gz` 合并进一条 batched reciprocal field 路径，减少了重复 transpose / Allgather 次数
- 如果你打开 `--test-phys-compute` 或使用 `compute mff/torch/phys`，physical tensor 会按需缓存到 host，这会额外增加输出搬运开销
- `--compare-throughput` 会自动在相同随机体系上跑两次：一次 baseline，一次打开你选定的长程/feature-FFT 模型路径，然后汇总 `loop time`、`steps/s` 和相对 slowdown
- 当前 `boundary p p f` reciprocal 路径已经单独走 `slab_2d` backend：它会在非周期方向使用 vacuum padding，再做正确性优先的频域求解
- `--test-pbc-slab` / `--test-feature-spectral-fft-slab` 检查横向跨边界对与盒内等价对是否一致
- `--test-pbc-slab-z-open` / `--test-feature-spectral-fft-slab-z-open` 检查远距 z 对不会被误当成最小像近邻
- 这仍然应视为 correctness-first 的首版 slab backend，不应与最终高性能/高精度 slab Green's function 实现画等号

**LAMMPS `in` 文件模板**：

下面假设：

- `core.pt` 已经导出完成
- 元素顺序为 `H O`
- cutoff 为 `5.0`
- 使用 `read_data system.data`

通用头部为：

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin
```

1. **无外场、无物理张量**：只输出 energy/force

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin

pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

fix 1 all nve
thermo_style custom step pe
thermo 20
run 200
```

2. **有外场、无物理张量**：以 rank-1 电场为例

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin

variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01

pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/core.pt H O

fix 1 all nve
thermo_style custom step pe
thermo 20
run 200
```

3. **无外场、有物理张量**：输出 energy/force/tensor

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin

pair_style mff/torch 5.0 cuda
pair_coeff * * /path/to/core.pt H O

compute mffgm all mff/torch/phys global/mask
compute mffdx all mff/torch/phys global dipole x
compute mffpxx all mff/torch/phys global polarizability xx
compute mffadx all mff/torch/phys atom dipole x

fix 1 all nve
thermo_style custom step pe c_mffgm[2] c_mffgm[3] c_mffdx c_mffpxx
dump 1 all custom 20 dump.tensor id type x y z c_mffadx
thermo 20
run 200
```

4. **有外场、有物理张量**：同时测试外场输入与 tensor 输出

```lammps
units metal
atom_style atomic
boundary p p p

read_data system.data

neighbor 1.0 bin

variable Ex equal 0.0
variable Ey equal 0.0
variable Ez equal 0.01

pair_style mff/torch 5.0 cuda field v_Ex v_Ey v_Ez
pair_coeff * * /path/to/core.pt H O

compute mffgm all mff/torch/phys global/mask
compute mffdx all mff/torch/phys global dipole x
compute mffpxx all mff/torch/phys global polarizability xx
compute mffadx all mff/torch/phys atom dipole x

fix 1 all nve
thermo_style custom step pe c_mffgm[2] c_mffgm[3] c_mffdx c_mffpxx
dump 1 all custom 20 dump.tensor id type x y z c_mffadx
thermo 20
run 200
```

**常用 `compute mff/torch/phys` 读法**：

```lammps
compute q all mff/torch/phys global charge
compute dx all mff/torch/phys global dipole x
compute d all mff/torch/phys global dipole
compute pxx all mff/torch/phys global polarizability xx
compute p all mff/torch/phys global polarizability

compute qatom all mff/torch/phys atom charge
compute dxatom all mff/torch/phys atom dipole x
compute datom all mff/torch/phys atom dipole
compute pxxatom all mff/torch/phys atom polarizability xx
compute patom all mff/torch/phys atom polarizability
```

其中 `global/mask` 的固定顺序为：

- `c_mffgm[1]` -> `charge`
- `c_mffgm[2]` -> `dipole`
- `c_mffgm[3]` -> `polarizability`
- `c_mffgm[4]` -> `quadrupole`

若某个量没有训练或没有导出，则对应 mask 为 `0`，输出块自动填 `0`。

**spherical-save-cue 导出说明**：默认导出为纯 PyTorch 实现（`force_naive`），`core.pt` 不依赖 cuEquivariance 自定义 ops，可在任意 LibTorch 环境运行。

完整说明见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)。

### 示例 5b: 使用 LAMMPS ML-IAP unified 接口

ML-IAP unified 是 LAMMPS 的机器学习势接口，相比 `fix external` 方式速度更快（约 1.7x），且支持 GPU 加速。使用前需将 checkpoint 导出为 ML-IAP 格式。

**模型限制**：仅以下五种模型支持 ML-IAP（因其支持 `precomputed_edge_vec`）：`e3nn_layers`、`e3nn_layers_channelwise`、`cue_layers_channelwise`（spherical-save-cue）、`pure_cartesian_ictd_layers`、`pure_cartesian_ictd_layers_full`。其他模型（如 pure-cartesian、pure-cartesian-sparse）暂不支持。

**步骤 1：导出模型**

```bash
python -m molecular_force_field.cli.export_mliap your_checkpoint.pth \
    --elements H O \
    --atomic-energy-keys 1 8 \
    --atomic-energy-values -13.6 -75.0 \
    --output model-mliap.pt
```

若 checkpoint 训练时已启用 ZBL，则导出的 `model-mliap.pt` 会自动继承相同的 ZBL 短程排斥修正；不需要在导出命令中再次显式传入 `--zbl-*`。

**自动 TorchScript 行为（重要）**：

- 对 `spherical-save-cue`，`python -m molecular_force_field.cli.export_mliap` 会**自动启用 TorchScript 导出**，即使用户没有显式传 `--torchscript`
- 这样做是为了避免该模式在普通 Python pickle 路径下出现不稳定或无法序列化的问题
- 因此你可以直接使用上面的命令；若 checkpoint 对应模式是 `spherical-save-cue`，CLI 会自动切换到安全导出路径
- 对 `pure-cartesian-ictd`、`pure-cartesian-ictd-save`，你仍可按需显式传 `--torchscript`
- `export_mliap` 也会优先从 checkpoint 自动恢复模型结构超参数；如果显式传了冲突的 CLI 参数，则**以 CLI 为准**
- 对新 checkpoint，`export_mliap` 也会优先读取 checkpoint 中保存的 `atomic_energy_keys/atomic_energy_values`；如果显式传了 `--atomic-energy-keys/--atomic-energy-values`，则以 CLI 为准
- 老 checkpoint 若没有保存 E0，则仍按旧逻辑回退到本地 `fitted_E0.csv`
- 这不会影响已有导出功能，只是让“默认不手填结构参数”变成安全可用的路径

**步骤 2：在 Python 中驱动 LAMMPS**

```python
import torch
import lammps
from lammps.mliap import activate_mliappy, load_unified

# 启动 LAMMPS 并激活 ML-IAP Python 模块
lmp = lammps.lammps()
activate_mliappy(lmp)

# 加载导出的模型
model = torch.load("model-mliap.pt", weights_only=False)
load_unified(model)

# 设置 LAMMPS 输入
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

**步骤 3：纯 LAMMPS 输入文件方式**

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

同样地，ML-IAP 运行时没有额外的 ZBL 关键字；只要 `model-mliap.pt` 是从启用 ZBL 的 checkpoint 导出的，运行时就会自动生效。

`input.lammps` 内容示例：

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

运行：

```bash
export DYLD_LIBRARY_PATH="$HOME/.local/lib:/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib"
python run.py
```

**注意事项：**

- `pair_coeff * * H O` 中元素顺序必须与导出时 `--elements H O` 一致
- `pair_style mliap unified model.pt 0` 末尾的 `0` 表示不包含 ghost 邻居；若模型有多层消息传递，可改为 `1`
- 使用 `units metal`（eV, Angstrom），与模型内部单位一致
- macOS 上需设置 `DYLD_LIBRARY_PATH` 指向 LAMMPS 和 Python 共享库
- LAMMPS 需编译时开启 `PKG_ML-IAP=ON`、`MLIAP_ENABLE_PYTHON=ON`，详见 `molecular_force_field/docs/INSTALL_LAMMPS_PYTHON.md`

**Kokkos GPU 加速**：若 LAMMPS 已用 Kokkos+CUDA 编译，可直接运行 `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half -in input.lammps`；Python 驱动时改用 `activate_mliappy_kokkos(lmp)`。详见 `LAMMPS_INTERFACE.md`。

### 示例 5c: 大体系多 GPU 推理（inference_ddp）

对于超大规模体系（如 10 万原子以上），可使用 DDP 推理进行多 GPU 并行计算。**仅支持 `pure-cartesian-ictd` 模式**。当前版本使用随机图进行测试；实际结构需在代码中接入。

```bash
torchrun --nproc_per_node=2 -m molecular_force_field.cli.inference_ddp \
  --checkpoint model.pth \
  --atoms 100000 \
  --forces
```

**参数说明**：
- `--atoms`: 原子数（用于生成随机测试图，默认 50000）
- `--checkpoint`: 模型检查点路径
- `--forces`: 同时计算并输出力
- `--partition`: 图分区策略，`modulo`（默认）或 `spatial`

### 示例 5d: 热导率工作流（MLFF -> IFC2/IFC3 -> intrinsic BTE -> Callaway）

这一工作流用于**晶体体系**的热导率计算，不使用 Green-Kubo 积分。推荐路线是：

1. 用 MLFF 计算位移超胞受力，生成 `IFC2/IFC3`
2. 用 `phono3py` 求解本征晶格热导率（intrinsic BTE）
3. 在 intrinsic BTE 结果上叠加 `Callaway` 风格的工程散射项，用于晶粒、缺陷、界面等工艺因素扫描

这条路线与 `mff-evaluate --phonon` 的定位不同：

- `mff-evaluate --phonon`：更适合做 Hessian、虚频、稳定性与频率分布检查
- `thermal_transport bte`：更适合做真正的热输运求解

#### 安装依赖

```bash
pip install -e ".[thermal]"
```

或手动安装：`pip install phonopy phono3py spglib scipy`

#### 入口

```bash
python -m molecular_force_field.cli.thermal_transport --help
```

提供两个子命令：

- `bte`：生成 `IFC2/IFC3` 并运行 intrinsic BTE
- `callaway`：在 `phono3py` 的 intrinsic 结果上做工程散射后处理

#### 步骤 1：运行 intrinsic BTE

```bash
python -m molecular_force_field.cli.thermal_transport bte \
  --checkpoint best_model.pth \
  --structure relaxed.cif \
  --supercell 4 4 4 \
  --phonon-supercell 4 4 4 \
  --mesh 16 16 16 \
  --temperatures 300 400 500 600 700 \
  --output-dir thermal_bte \
  --device cuda \
  --atomic-energy-file fitted_E0.csv
```

**工作流程**：

1. 从 checkpoint 自动恢复模型结构超参数与 `tensor_product_mode`
2. 读取晶体结构，必要时可用 `--relax-fmax` 先用 MLFF 预优化
3. 调用 `phono3py` 生成 FC2/FC3 位移超胞
4. 通过现有 ASE calculator 批量计算每个位移超胞的受力
5. 生成 `fc2.hdf5`、`fc3.hdf5`
6. 运行 intrinsic BTE，默认是 `RTA`；使用 `--lbte` 可切换为迭代 `LBTE`

**主要输出文件**：

- `fc2.hdf5`
- `fc3.hdf5`
- `fc2_forces.hdf5`
- `fc3_forces.hdf5`
- `fc2_forces.txt`
- `fc3_forces.txt`
- `thermal_workflow_metadata.json`
- `kappa-*.hdf5`

**推荐实践**：

- 输入结构最好已经接近平衡相
- 先用 `mff-evaluate --phonon` 检查有没有明显虚频
- 对 `--supercell`、`--phonon-supercell`、`--mesh` 做收敛测试
- 初次试算先用 `RTA`，稳定后再切 `LBTE`

#### 步骤 2：在 intrinsic BTE 上叠加 Callaway 工程散射

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway \
  --component xx \
  --grain-size-nm 200 \
  --point-defect-coeff 1.0e-4
```

当前后处理使用 `phono3py` 的模态热导率与线宽，在此基础上施加 Matthiessen 叠加：

- 边界散射：由 `grain_size_nm` 和 `specularity` 控制
- 点缺陷散射：`point_defect_coeff * omega^4`
- 位错散射：`dislocation_coeff * omega^2`
- 界面散射：`interface_coeff * omega`

**输出文件**：

- `<prefix>.csv`
- `<prefix>.json`
- `<prefix>_summary.json`

#### 步骤 3：用实验热导率反推工程散射参数

若已有实验热导率，可只拟合外禀散射项，而不必重复跑 BTE：

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway_fit \
  --fit-experiment-csv exp_kappa.csv \
  --fit-component xx \
  --fit-parameters grain_size_nm,point_defect_coeff
```

实验 CSV 至少需要：

- `temperature`
- 一个热导率列，如 `xx`、`yy`、`zz`，或通过 `--fit-column` 指定的列

这一步的物理含义是：

- intrinsic BTE 保持不变
- 只拟合晶粒、缺陷、界面等外禀散射参数
- 拟合好的参数可用于不同工艺窗口下的快速扫描

#### 工程上推荐的使用顺序

1. 先验证 MLFF 在平衡结构附近的声子稳定性
2. 用 `bte` 得到本征 `kappa(T)`
3. 用实验数据只拟合 Callaway 外禀散射参数
4. 再用拟合好的参数扫描晶粒尺寸、缺陷浓度、界面质量
5. 最后把得到的 `k(T)` 曲线导入 COMSOL、ANSYS 或器件级热仿真流程

更完整的热导率文档见仓库根目录的 `THERMAL_TRANSPORT.md`。

### 示例 6: 声子谱计算（Hessian 和频率）

```python
from ase.io import read
from ase.optimize import BFGS
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import neighbor_list
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.evaluation.evaluator import Evaluator
from molecular_force_field.models import E3_TransformerLayer_multi
from molecular_force_field.utils.config import ModelConfig
import torch
import numpy as np

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load('combined_model.pth', map_location=device)

config = ModelConfig()
config.load_atomic_energies_from_file('fitted_E0.csv')

e3trans = E3_TransformerLayer_multi(...).to(device)
e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
e3trans.eval()

# 创建计算器和评估器
ref_energies_dict = {
    k.item(): v.item()
    for k, v in zip(config.atomic_energy_keys, config.atomic_energy_values)
}
calc = MyE3NNCalculator(e3trans, ref_energies_dict, device, max_radius=5.0)

class SimpleDataset:
    def restore_force(self, x):
        return x
    def restore_energy(self, x):
        return x

dataset = SimpleDataset()
evaluator = Evaluator(
    model=e3trans,
    dataset=dataset,
    device=device,
    atomic_energy_keys=config.atomic_energy_keys,
    atomic_energy_values=config.atomic_energy_values,
)

# 读取结构
atoms = read('structure.xyz')
atoms.calc = calc

# 可选：优化结构
print("Relaxing structure...")
optimizer = BFGS(atoms, logfile=None)
optimizer.run(fmax=0.01)
print(f"Relaxation completed. Final forces: max={atoms.get_forces().max():.6f} eV/Å")

# 准备数据
pos = torch.tensor(atoms.get_positions(), dtype=torch.get_default_dtype(), device=device)
A = torch.tensor([atomic_numbers[symbol] for symbol in atoms.get_chemical_symbols()], 
                 dtype=torch.long, device=device)
batch_idx = torch.zeros(len(atoms), dtype=torch.long, device=device)

# 构建图（使用 ASE on-the-fly 邻居列表）
cell_array = atoms.cell.array
pbc_flags = atoms.pbc
if cell_array is None or np.abs(cell_array).sum() <= 1e-9:
    cell_array = np.eye(3) * 100.0
    pbc_flags = [False, False, False]

atoms_nl = atoms.copy()
atoms_nl.cell = cell_array
atoms_nl.pbc = pbc_flags

edge_src_np, edge_dst_np, edge_shifts_np = neighbor_list('ijS', atoms_nl, max_radius=5.0)
edge_src = torch.tensor(edge_src_np, dtype=torch.long, device=device)
edge_dst = torch.tensor(edge_dst_np, dtype=torch.long, device=device)
edge_shifts = torch.tensor(edge_shifts_np, dtype=torch.float64, device=device)
cell = torch.tensor(cell_array, dtype=torch.get_default_dtype(), device=device).unsqueeze(0)

# 计算 Hessian 矩阵
print("Computing Hessian matrix...")
hessian = evaluator.compute_hessian(
    pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell
)

print(f"Hessian shape: {hessian.shape}")
print(f"Hessian range: [{hessian.min():.6f}, {hessian.max():.6f}] eV/Å²")

# 检查对称性
is_symmetric = torch.allclose(hessian, hessian.T, atol=1e-4)
print(f"Hessian symmetric: {is_symmetric}")

# 获取原子质量
masses = torch.tensor([atomic_masses[atomic_numbers[symbol]] 
                      for symbol in atoms.get_chemical_symbols()],
                     dtype=torch.get_default_dtype())

# 计算声子谱
print("Computing phonon spectrum...")
frequencies = evaluator.compute_phonon_spectrum(
    hessian, masses, output_prefix='phonon'
)

print(f"\nPhonon calculation completed!")
print(f"Total modes: {len(frequencies)}")
print(f"Real modes: {np.sum(frequencies >= 0)}")
print(f"Imaginary modes: {np.sum(frequencies < 0)}")
print(f"Frequency range: [{frequencies.min():.2f}, {frequencies.max():.2f}] cm⁻¹")

# 分析结果
if np.any(frequencies < 0):
    print(f"\n⚠️  Warning: {np.sum(frequencies < 0)} imaginary frequencies detected!")
    print("   This indicates the structure may not be at a local minimum.")
    print("   Consider further optimization or checking the structure.")
else:
    print("\n✅ All frequencies are real (structure is stable).")
```

## 常见问题

### Q: 如何查看所有可用的命令行参数？

```bash
mff-preprocess --help
mff-train --help
mff-evaluate --help
mff-export-core --help   # LAMMPS LibTorch 导出
mff-lammps --help        # LAMMPS fix external 接口
python -m molecular_force_field.cli.export_mliap --help  # LAMMPS ML-IAP 导出
```

### Q: 如何使用 LAMMPS LibTorch 接口？

LAMMPS LibTorch 接口（USER-MFFTORCH）通过 `pair_style mff/torch` 在 C++ 侧用 LibTorch 加载 TorchScript 模型，**运行时无需 Python**，适合 HPC 与生产部署。

**快速步骤**：
1. 导出：`mff-export-core --checkpoint model.pth --elements H O --e0-csv fitted_E0.csv --out core.pt`
2. 编译 LAMMPS：启用 `PKG_KOKKOS`、`PKG_USER-MFFTORCH`，见 [lammps_user_mfftorch/docs/BUILD_AND_RUN.md](lammps_user_mfftorch/docs/BUILD_AND_RUN.md)
3. 运行：`lmp -k on g 1 -sf kk -pk kokkos newton off neigh full -in in.mfftorch`

**支持模型**：`pure-cartesian-ictd` 系列、`spherical-save-cue`。完整说明见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)。

### Q: 如何导出 ML-IAP 格式？

ML-IAP 用于 LAMMPS 的 `pair_style mliap unified`，比 fix external 更快且支持 Kokkos GPU：

```bash
python -m molecular_force_field.cli.export_mliap checkpoint.pth \
  --elements H O --atomic-energy-keys 1 8 --atomic-energy-values -13.6 -75.0 \
  --max-radius 5.0 --output model-mliap.pt
```

支持模型：`spherical`、`spherical-save`、`spherical-save-cue`、`pure-cartesian-ictd`、`pure-cartesian-ictd-save`。详见 [LAMMPS_INTERFACE.md](LAMMPS_INTERFACE.md)。

补充说明：

- 对 `spherical-save-cue`，CLI 会自动启用 TorchScript 导出，无需手动添加 `--torchscript`
- 不支持 `pure-cartesian`、`pure-cartesian-sparse`

### Q: 如何计算热导率？

推荐的晶体热导率路线是：

1. `MLFF -> IFC2/IFC3`
2. `IFC2/IFC3 -> intrinsic BTE`
3. `intrinsic BTE -> Callaway 工程散射`

入口：

```bash
python -m molecular_force_field.cli.thermal_transport --help
```

典型命令：

```bash
python -m molecular_force_field.cli.thermal_transport bte \
  --checkpoint best_model.pth \
  --structure relaxed.cif \
  --supercell 4 4 4 \
  --phonon-supercell 4 4 4 \
  --mesh 16 16 16 \
  --temperatures 300 400 500 600 700 \
  --output-dir thermal_bte \
  --device cuda \
  --atomic-energy-file fitted_E0.csv
```

```bash
python -m molecular_force_field.cli.thermal_transport callaway \
  --kappa-hdf5 thermal_bte/kappa-m161616.hdf5 \
  --output-prefix thermal_bte/callaway \
  --component xx \
  --grain-size-nm 200 \
  --point-defect-coeff 1.0e-4
```

详细说明见 `THERMAL_TRANSPORT.md`。

### Q: 训练时如何恢复之前的检查点？

训练器会**自动检测并加载**检查点文件。如果 `--checkpoint` 指定的文件已存在，会自动恢复训练状态。

**自动恢复的内容：**
- ✅ 模型权重（`e3trans_state_dict`）
- ✅ Epoch 数（从中断的 epoch 继续）
- ✅ Batch count（累计的 batch 数）
- ✅ 损失权重 `a` 和 `b`
- ✅ 最佳验证损失（用于早停判断，基于 `a × 能量损失 + b × 受力损失 + c × 应力损失`）
- ✅ 早停计数器（patience counter）
- ✅ 模型结构超参数（优先级：`CLI 显式参数 > checkpoint.model_hyperparameters / checkpoint 顶层字段 > 默认值`）

**推荐用法：**

- 恢复训练时，通常只需要继续传 `--checkpoint` 和训练运行参数；大多数模型结构超参数可以省略
- 只有在你**明确想覆盖 checkpoint 中的结构配置**时，才显式传 `--embedding-dim`、`--output-size`、`--lmax`、`--num-interaction`、`--max-radius`、`--tensor-product-mode`、`--invariant-channels` 等参数
- 如果 CLI 和 checkpoint 冲突，当前行为是**CLI 覆盖 checkpoint**
- 对非常老、未保存 `model_hyperparameters` 的 checkpoint，仍可能需要手动补部分参数

**使用示例：**

```bash
# 第一次训练（训练到 epoch 50 时中断）
mff-train \
    --data-dir data \
    --epochs 100 \
    --checkpoint my_model.pth \
    --device cuda

# 程序中断后，再次运行相同命令即可继续
# 会自动从 epoch 51 继续训练
mff-train \
    --data-dir data \
    --epochs 100 \
    --checkpoint my_model.pth \
    --device cuda

# 日志会显示：
# ================================================================================
# Checkpoint Loaded Successfully!
#   Resuming from epoch: 51
#   Batch count: 12500
#   Loss weights: a=1.0234, b=9.7845
#   Best validation loss: 0.012345
#   Early stopping patience counter: 3/20
# ================================================================================
```

**注意事项：**
1. 若你没有显式覆盖，训练会优先从 checkpoint 自动恢复模型结构超参数
2. 确保使用**相同的数据目录**和**相同的设备**
3. 可以调整 `--epochs` 来延长训练（例如从 100 改为 200）
4. **不能**在恢复时改变学习率调度器的初始设置（会重新初始化）

### Q: 现在 `mff-evaluate` 应该怎么用？

现在推荐优先使用“最小命令”：

```bash
mff-evaluate \
    --checkpoint best_model.pth \
    --test-prefix test \
    --output-prefix test \
    --use-h5
```

说明：

- `mff-evaluate` 会优先从 checkpoint 自动恢复 `tensor_product_mode` 和模型结构超参数
- 对新 checkpoint，`mff-evaluate` 也会优先读取 checkpoint 中保存的 `atomic_energy_keys/atomic_energy_values`
- 如果你显式传了冲突参数，例如 `--tensor-product-mode`、`--embedding-dim`、`--output-size`，则**以 CLI 为准**
- 如果你显式传了 `--atomic-energy-file` 或 `--atomic-energy-keys/--atomic-energy-values`，这些 E0 输入也会覆盖 checkpoint
- 老 checkpoint 若没有保存 E0，则仍按旧逻辑读取本地 `fitted_E0.csv`
- 因此，只有在你明确想做覆盖测试时，才建议手动传这些结构参数

**如何从头重新训练（忽略已有检查点）：**
```bash
# 方法1: 使用不同的检查点名称
mff-train --checkpoint new_model.pth ...

# 方法2: 删除或重命名旧检查点
mv my_model.pth my_model_backup.pth
mff-train --checkpoint my_model.pth ...
```

### Q: 如何调整模型超参数？

可以通过修改 `molecular_force_field/utils/config.py` 中的 `ModelConfig` 类，或者在使用Python API时直接传入参数。

### Q: 如何调整模型超参数？

可以通过CLI参数调整模型架构：

**原子嵌入参数：**
```bash
# 默认设置（适用于H、C、N、O等常见元素）
mff-train --max-atomvalue 10 --embedding-dim 16

# 包含更多元素类型（如金属）
mff-train --max-atomvalue 50 --embedding-dim 32

# 增大嵌入维度以提升模型表达能力
mff-train --max-atomvalue 10 --embedding-dim 32
```

**模型容量参数（lmax 和通道数）：**
```bash
# 默认设置（lmax=2, 64通道, gaussian基函数）
# 生成: "64x0e + 64x1o + 64x2e"
mff-train --lmax 2 --irreps-output-conv-channels 64

# 降低最高阶数（更简单的体系，更快训练）
# 生成: "64x0e + 64x1o"
mff-train --lmax 1 --irreps-output-conv-channels 64

# 增加最高阶数（更复杂的体系，更强表达能力）
# 生成: "64x0e + 64x1o + 64x2e + 64x3o"
mff-train --lmax 3 --irreps-output-conv-channels 64

# 增大通道数（更大模型容量）
# 生成: "128x0e + 128x1o + 128x2e"
mff-train --lmax 2 --irreps-output-conv-channels 128

# 同时增大lmax和通道数（最大模型容量，适合复杂体系）
# 生成: "128x0e + 128x1o + 128x2e + 128x3o"
mff-train --lmax 3 --irreps-output-conv-channels 128

# 减小模型容量（适合简单体系或内存受限）
# 生成: "32x0e + 32x1o"
mff-train --lmax 1 --irreps-output-conv-channels 32
```

**径向基函数类型：**
```bash
# 默认：高斯基函数（推荐，平滑且易于优化）
mff-train --function-type gaussian

# 贝塞尔基函数（适合周期性系统）
mff-train --function-type bessel

# 傅里叶基函数（适合周期性边界条件）
mff-train --function-type fourier

# 余弦基函数
mff-train --function-type cosine

# 平滑有限支撑基函数
mff-train --function-type smooth_finite
```

**注意事项：**
- `--max-atomvalue` 必须大于等于数据集中最大的原子序数
- `--lmax` 控制球谐函数的最高阶数，影响模型对角度信息的表达能力
  - `lmax=0`: 仅标量 (scalar)
  - `lmax=1`: 标量 + 矢量 (scalar + vector)
  - `lmax=2`: 标量 + 矢量 + 二阶张量 (scalar + vector + rank-2 tensor)
  - `lmax=3`: 包含三阶张量
  - 更高的 lmax 可以捕捉更复杂的几何特征，但计算量显著增加
- `--function-type` 选择径向基函数类型：
  - `gaussian`: 最常用，适合大多数体系
  - `bessel`: 适合具有周期性的体系（如晶体）
  - `fourier`: 适合周期性边界条件
  - 通常默认的 `gaussian` 就足够了
- 增大 `--embedding-dim`、`--lmax` 和 `--irreps-output-conv-channels` 会显著增加显存占用和训练时间
- 评估时必须使用与训练时相同的超参数值
- 这些参数会影响模型性能，建议根据数据集规模和复杂度调整

**完整示例：大模型配置**
```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --max-atomvalue 20 \
    --embedding-dim 32 \
    --lmax 3 \
    --irreps-output-conv-channels 128 \
    --function-type gaussian \
    --batch-size 4 \
    --device cuda
```

### Q: 如何调整学习率？

训练时可以通过以下参数调整学习率策略：

**基础学习率设置：**
```bash
mff-train \
    --learning-rate 2e-4 \      # 目标学习率（预热后达到的值）
    --min-learning-rate 1e-5    # 最小学习率（防止过小）
```

**学习率预热（Warmup）：**
```bash
mff-train \
    --warmup-batches 1000 \     # 前1000个batch进行预热
    --warmup-start-ratio 0.1    # 从10%的目标学习率开始，线性增长到100%
```
预热期间，学习率从`learning_rate × warmup_start_ratio`线性增加到`learning_rate`。例如，如果`--learning-rate 2e-4`和`--warmup-start-ratio 0.1`，则学习率会从`2e-5`线性增长到`2e-4`。这有助于训练初期的稳定性。

**学习率衰减（Decay）：**
```bash
mff-train \
    --lr-decay-patience 1000 \  # 每1000个batch检查一次，如果没有改善则衰减
    --lr-decay-factor 0.98      # 每次衰减时学习率乘以0.98
```
例如：初始学习率2e-4，每1000个batch后如果验证指标没有改善，学习率变为2e-4 × 0.98 = 1.96e-4，再1000个batch后变为1.96e-4 × 0.98 = 1.92e-4，以此类推。

**完整示例：**
```bash
# 快速收敛设置（较大学习率，快速预热）
mff-train \
    --learning-rate 5e-4 \
    --min-learning-rate 1e-5 \
    --warmup-batches 500 \
    --warmup-start-ratio 0.1

# 稳定训练设置（较小学习率，更长预热）
mff-train \
    --learning-rate 1e-4 \
    --min-learning-rate 1e-6 \
    --warmup-batches 2000 \
    --warmup-start-ratio 0.05

# 精细调优设置（小学习率，慢衰减）
mff-train \
    --learning-rate 1e-5 \
    --lr-decay-factor 0.99 \
    --lr-decay-patience 2000 \
    --warmup-start-ratio 0.2
```

### Q: 如何调整损失权重？

损失函数为：`总损失 = a × 能量损失 + b × 受力损失 + c × 应力损失`（当 `-c > 0` 时包含应力项）

**初始权重设置：**
```bash
# 默认设置（能量权重1.0，受力权重10.0，应力权重0即禁用）
mff-train -a 1.0 -b 10.0

# 更重视能量（如果能量预测较差）
mff-train -a 2.0 -b 5.0

# 更重视受力（如果受力预测较差）
mff-train -a 0.5 -b 20.0

# 启用应力训练（需要 XYZ 含 stress/virial，且为 PBC 体系）
mff-train -a 1.0 -b 10.0 -c 0.1
```

**自动调整权重：**
```bash
# 每500个batch自动调整一次（更频繁）
mff-train -a 1.0 -b 10.0 --update-param 500

# 每2000个batch自动调整一次（更稳定）
mff-train -a 1.0 -b 10.0 --update-param 2000
```

**调整速率控制：**
```bash
# 慢速调整（适合超长时间训练，200k+ batches）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.005 --weight-b-decay 0.995

# 中速调整（更稳定，适合大多数情况）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.01 --weight-b-decay 0.99

# 快速调整（适合短期训练，<50k batches）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.02 --weight-b-decay 0.98

# 超快速调整（默认）
mff-train -a 1.0 -b 10.0 --weight-a-growth 1.05 --weight-b-decay 0.98
```

自动调整规则：每`--update-param`个batch，`a`会乘以`--weight-a-growth`（增加），`b`会乘以`--weight-b-decay`（减少），以逐渐平衡能量和受力的重要性。**注意：`a` 和 `b` 默认会被限制在 [1, 1000]，以防训练过长导致权重漂移过大。** 应力权重 `c` 可通过 `--c-min` 和 `--c-max` 限制范围。

**可选：为 a / b 设置范围（Clamp）**（适合长训练防止权重漂移过大）：
```bash
# 将 a 限制在 [0.5, 10]，b 限制在 [0.001, 100]
mff-train \
  -a 1.0 -b 10.0 \
  --a-min 0.5 --a-max 10 \
  --b-min 0.001 --b-max 100
```

**速率选择建议：**
- **慢速 (1.005/0.995)**: 适合超长时间训练（>200k batches），权重变化平缓，训练更稳定
- **中速 (1.01/0.99)**: 适合大多数训练场景（50k-200k batches），平衡稳定性和调整速度
- **快速 (1.02/0.98)**: 适合短期训练（<50k batches），快速调整权重，但可能导致训练后期不稳定
- **默认 (1.05/0.98)**: 调整更激进，收敛更快但更容易不稳定；如果出现震荡，建议改用中速或慢速

### Q: 如何进行应力训练？

应力训练用于周期性体系（PBC），通过晶胞应变导数计算应力 σ = (1/V) × dE/dε，与参考应力做 MSE 作为 stress_loss。

**前提条件：**
1. XYZ 文件为周期性体系（含 `pbc="T T T"` 和 `Lattice="..."`）
2. XYZ comment 行含 stress 或 virial，例如：`stress="0.01 0 0 0 0.01 0 0 0 0.01"`（9 个 3×3 分量，eV/Å³）

**启用应力训练：**
```bash
mff-train -a 1.0 -b 10.0 -c 0.1 --input-file pbc_with_stress.xyz
```

**权重建议：** 应力分量少（每结构 9 个），通常 `-c` 取 0.01～0.5。若应力预测偏差大，可适当增大 `-c`。

### Q: 如何调整优化器参数？

**梯度裁剪（防止梯度爆炸）：**
```bash
# 更严格的梯度裁剪（适合训练不稳定时）
mff-train --max-grad-norm 0.3

# 更宽松的梯度裁剪（适合训练稳定时）
mff-train --max-grad-norm 1.0
```

**优化器稳定性参数：**
```bash
# 更频繁地钳位v_hat（适合训练不稳定时）
mff-train --vhat-clamp-interval 1000 --max-vhat-growth 3.0

# 更宽松的v_hat控制（适合训练稳定时）
mff-train --vhat-clamp-interval 5000 --max-vhat-growth 10.0
```

**监控梯度：**
```bash
# 每100个batch记录一次梯度统计（用于调试）
mff-train --grad-log-interval 100

# 每1000个batch记录一次（减少日志量）
mff-train --grad-log-interval 1000
```

### Q: 如何选择合适的批次大小？

批次大小影响训练速度和内存使用：

```bash
# 小批次（适合GPU内存较小或数据集较小）
mff-train --batch-size 4

# 中等批次（默认，适合大多数情况）
mff-train --batch-size 8

# 大批次（适合GPU内存充足，可以加速训练）
mff-train --batch-size 16
```

**注意：** 批次大小会影响梯度估计的稳定性。如果批次太小（如2或4），可能需要降低学习率；如果批次太大，可能需要增加学习率。

### Q: 如何调整验证和保存频率？

```bash
# 更频繁的验证和保存（适合快速迭代）
mff-train --dump-frequency 100

# 默认频率（平衡性能和监控）
mff-train --dump-frequency 250

# 较少验证（适合长时间训练，减少I/O开销）
mff-train --dump-frequency 500
```

**注意：** `--dump-frequency`控制验证和模型保存的频率。更频繁的验证可以更早发现问题，但会增加训练时间。

### Q: 数据格式要求是什么？

输入的XYZ文件需要包含：
- 原子坐标 (x, y, z)
- 原子类型/原子序数
- 力 (Fx, Fy, Fz)
- 能量信息（在Properties行中）
- 晶胞信息（在Lattice属性中，可选）
- **应力/维里张量**（可选，用于周期性体系的应力训练）：在 comment 行中支持以下格式：
  - `stress="xx yy zz yz xz xy"` — 6 个 Voigt 分量 (eV/Å³)
  - `stress="xx xy xz yx yy yz zx zy zz"` — 9 个 3×3 分量，行优先 (eV/Å³)
  - `virial="xx xy xz yx yy yz zx zy zz"` — 9 个维里分量 (eV)，会自动转换为 stress = -virial/V

### Q: 如何使用多卡并行训练？

多卡训练可以显著加速训练过程，特别是在大数据集上。

**方式 1：直接使用 `--n-gpu`（推荐，最简单）：**

```bash
# 4 卡训练，自动启动 torchrun DDP
mff-train --n-gpu 4 --data-dir data --epochs 1000 --batch-size 8

# 多节点：2 节点 × 4 卡（SLURM 环境下自动检测）
mff-train --n-gpu 4 --nnodes 2 --data-dir data --epochs 1000
```

> `--n-gpu 1`（默认）时行为完全不变，与原始 `mff-train` 完全兼容。
> `--n-gpu N` (N > 1) 或 `--nnodes > 1` 时，自动重启为 `torchrun --nproc_per_node=N --distributed ...`。
> 原有的 `torchrun ... --distributed` 手动用法依然完全支持。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--n-gpu` | 1 | GPU 数。>1 自动 torchrun DDP |
| `--nnodes` | 1 | 节点数。>1 多节点 DDP |
| `--master-addr` | auto | rendezvous 地址 |
| `--master-port` | 29500 | rendezvous 端口 |
| `--launcher` | auto | auto / local / slurm |

**方式 2：手动 torchrun（完全向后兼容）：**

```bash
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**注意事项：**
- 每个GPU的有效batch size = `--batch-size`，总的有效batch size = `--batch-size × GPU数量`
- 建议根据总的有效batch size调整学习率（通常线性缩放）
- 所有日志、检查点和CSV文件只在主进程（rank 0）保存
- 验证结果会自动聚合所有进程的结果
- checkpoint 写入后所有 rank 通过 `dist.barrier()` 同步

**示例：4卡训练，每卡batch size=8，总有效batch size=32**
```bash
mff-train --n-gpu 4 --data-dir data --batch-size 8 --learning-rate 4e-3
```

### Q: 如何使用多卡并行训练？（旧版方式）

如果你使用的是旧版 PyTorch 或需要更细粒度的控制：

**使用 torchrun（推荐，PyTorch 1.9+）：**
```bash
# 使用4张GPU
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**2. 使用 torch.distributed.launch（旧版本兼容）：**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8
```

**注意事项：**
- `--nproc_per_node` 指定使用的GPU数量
- 必须添加 `--distributed` 参数
- 每个GPU的有效batch size = `--batch-size`，总的有效batch size = `--batch-size × GPU数量`
- 建议根据总的有效batch size调整学习率（通常线性缩放）
- 所有日志、检查点和CSV文件只在主进程（rank 0）保存
- 验证结果会自动聚合所有进程的结果

**示例：4卡训练，每卡batch size=8，总有效batch size=32**
```bash
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --batch-size 8 \
    --learning-rate 4e-3  # 可以适当增加学习率（4倍batch size）
```

### Q: 如何加速训练？

**1. 使用优化的张量积模式：**
```bash
# Pure-Cartesian-ICTD: 参数量最少（减少 72.1%），速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x）
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-ictd

# Pure-Cartesian-Sparse: 参数量减少 29.6%，速度稳定（CPU: 0.53x-1.39x，GPU: 0.46x-1.17x）
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-sparse

# Partial-Cartesian-Loose: 非严格等变（norm product 近似），速度较快（CPU: 0.17x-1.37x，GPU: 0.21x-1.52x），参数量减少 17.3%
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian-loose

# Partial-Cartesian: 严格等变，参数量减少 17.4%
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian
```

**2. 使用预处理数据：**
```bash
# 预处理时使用 --preprocess-h5 预计算邻居列表
mff-preprocess --input-file 2000.xyz --preprocess-h5

# 训练时使用预处理的数据（自动检测）
mff-train --data-dir data
```

**3. 使用GPU：**
```bash
mff-train --device cuda
```

**4. 增加并行处理：**
```bash
# 增加数据加载的并行进程数
mff-train --num-workers 16

# 注意：num_workers会根据CPU核心数自动调整，训练时DataLoader会使用一半线程
```

**5. 使用float32（如果精度足够）：**
```bash
# float32比float64快约2倍，但精度略低
mff-train --dtype float32
```

**6. 增加批次大小（如果GPU内存允许）：**
```bash
# 更大的批次可以更好地利用GPU
mff-train --batch-size 16
```

**7. 减少验证频率（长时间训练时）：**
```bash
# 减少I/O开销
mff-train --dump-frequency 500
```

**8. 使用多卡并行训练（最有效）：**
```bash
# 使用4张GPU并行训练，速度提升接近4倍
torchrun --nproc_per_node=4 \
    -m molecular_force_field.cli.train \
    --distributed \
    --data-dir data \
    --batch-size 8
```

**9. 验证阶段使用 torch.compile（可选）：**
```bash
mff-train --compile-val e3trans --data-dir data
```

### Q: 应该选择哪种张量积模式？

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 发表论文/高精度需求 | `spherical` | 使用 e3nn 严格球谐，精度最高，标准实现 |
| 参数量优化（最多） | `pure-cartesian-ictd` | 参数量减少 72.1%，速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x），严格等变 |
| 参数量优化（平衡） | `pure-cartesian-sparse` | 参数量减少 29.6%，速度稳定（CPU: 0.53x-1.39x，GPU: 0.46x-1.17x），严格等变 |
| GPU 内存受限 | `pure-cartesian-ictd` 或 `pure-cartesian-sparse` | 参数量减少 72.1% 或 29.6%，显存占用更低 |
| 严格等变性要求 | `spherical`、`partial-cartesian`、`pure-cartesian-sparse` 或 `pure-cartesian-ictd` | 这些模式都严格等变 |
| 快速实验迭代 | `pure-cartesian-ictd` | 速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x），严格等变 |
| 首次尝试/对比基准 | `spherical` | 验证模型正确性，性能基准 |
| 大规模模型部署 | `pure-cartesian-ictd` | 参数量最少（减少 72.1%），速度最快（CPU: 最高 4.12x，GPU: 最高 2.10x） |

**切换模式：**
```bash
# 球谐模式（默认）
mff-train --input-file data.xyz

# 其他模式
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian
mff-train --input-file data.xyz --tensor-product-mode partial-cartesian-loose
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-sparse
mff-train --input-file data.xyz --tensor-product-mode pure-cartesian-ictd
```

**注意：** 训练和评估必须使用相同的模式！

### Q: 如何使用 SWA 和 EMA？

**SWA（Stochastic Weight Averaging）** 用于在训练后期固定损失权重，避免权重漂移：
```bash
# 从 epoch 100 开始，将 a 固定为 100，b 固定为 10
mff-train \
    --input-file data.xyz \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 100 --swa-a 100 --swa-b 10
```

**EMA（Exponential Moving Average）** 用于维护模型参数的滑动平均，提高稳定性：
```bash
# 从 epoch 150 开始启用 EMA，衰减系数 0.999
mff-train \
    --input-file data.xyz \
    --ema-start-epoch 150 --ema-decay 0.999 \
    --use-ema-for-validation \
    --save-ema-model
```

**同时使用 SWA 和 EMA：**
```bash
mff-train \
    --input-file data.xyz \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 100 --swa-a 100 --swa-b 10 \
    --ema-start-epoch 150 --ema-decay 0.999 --use-ema-for-validation --save-ema-model
```

**推荐设置：**
- SWA 开始时间：总 epoch 的 50%-70%
- EMA 开始时间：总 epoch 的 60%-80%
- EMA 衰减系数：0.999（默认，越大越平滑）

**注意：** SWA 启动时会重置早停计数器（patience_counter）和最佳验证损失（best_val_loss），以适应新的损失权重。

### Q: 训练时如何监控进度？

**控制台输出：**
- 验证结果会实时输出到控制台
- 每个epoch的摘要信息会输出到控制台
- 验证阶段每个batch的能量预测（由`--log-val-batch-energy`控制，默认False，只记录到日志文件）

**日志文件：**
- `training_YYYYMMDD_HHMMSS.log` - 包含所有详细信息
  - Batch级别的训练指标（每N个batch，由`--energy-log-frequency`控制；启用应力训练时还会输出 Stress Loss）
  - 验证时的详细结果（包括每个batch的能量预测，无论`--log-val-batch-energy`设置如何；启用应力训练时包含 Val Stress Loss/RMSE/MAE）
  - 梯度统计信息（每N个batch，由`--grad-log-interval`控制）

**CSV文件：**
- `training_YYYYMMDD_HH_loss.csv` - 训练损失记录
- `val_energy_epoch{epoch}_batch{batch_count}.csv` - 验证集能量预测
- `val_force_epoch{epoch}_batch{batch_count}.csv` - 验证集力预测

**实时监控：**
```bash
# 实时查看日志文件
tail -f training_*.log

# 查看最新的验证结果
tail -f training_*.log | grep "Validation"
```

## 完整训练示例

### 示例 1: 快速测试（小数据集，快速验证）

```bash
mff-train \
    --input-file small_test.xyz \
    --data-dir data \
    --epochs 100 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --warmup-batches 100 \
    --dump-frequency 50 \
    --device cuda \
    --dtype float32
```

**适用场景：** 快速验证代码和参数设置是否正确

### 示例 2: 标准训练（中等数据集，平衡设置）

```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --learning-rate 1e-3 \
    --min-learning-rate 2e-5 \
    --warmup-batches 1000 \
    --warmup-start-ratio 0.1 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.98 \
    -a 1.0 \
    -b 10.0 \
    --update-param 1000 \
    --patience 20 \
    --dump-frequency 250 \
    --device cuda \
    --dtype float64
```

**适用场景：** 大多数情况下的推荐配置

### 示例 3: 大规模训练（大数据集，长时间训练）

```bash
mff-train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --epochs 5000 \
    --batch-size 16 \
    --learning-rate 1e-3 \
    --min-learning-rate 5e-6 \
    --warmup-batches 2000 \
    --warmup-start-ratio 0.05 \
    --lr-decay-patience 2000 \
    --lr-decay-factor 0.99 \
    -a 1.0 \
    -b 10.0 \
    --update-param 2000 \
    --patience 50 \
    --dump-frequency 500 \
    --max-grad-norm 0.5 \
    --vhat-clamp-interval 2000 \
    --device cuda \
    --dtype float64 \
    --num-workers 16
```

**适用场景：** 大规模数据集，需要长时间训练以获得最佳性能

### 示例 4: 精细调优（接近收敛，小步调整）

```bash
mff-train \
    --data-dir data \
    --checkpoint combined_model.pth \
    --epochs 500 \
    --batch-size 8 \
    --learning-rate 5e-5 \
    --min-learning-rate 1e-6 \
    --warmup-batches 500 \
    --warmup-start-ratio 0.2 \
    --lr-decay-patience 1000 \
    --lr-decay-factor 0.995 \
    -a 0.5 \
    -b 15.0 \
    --update-param 2000 \
    --patience 30 \
    --dump-frequency 100 \
    --device cuda \
    --dtype float64
```

**适用场景：** 从已有检查点继续训练，进行精细调优

### 示例 5: CPU训练（无GPU环境）

```bash
mff-train \
    --input-file 2000.xyz \
    --data-dir data \
    --epochs 500 \
    --batch-size 4 \
    --learning-rate 1e-3 \
    --warmup-batches 500 \
    --dump-frequency 100 \
    --device cpu \
    --dtype float32 \
    --num-workers 4
```

**适用场景：** 在没有GPU的机器上训练，使用float32和较小的批次大小以加速

## 注意事项

1. **内存使用**: 使用 `H5Dataset` 时，数据会预加载到内存，确保有足够的内存。如果内存不足，可以考虑使用 `OnTheFlyDataset`（但训练速度会较慢）。

2. **GPU内存**: 如果遇到GPU内存不足（OOM错误），可以：
   - 减小 `--batch-size`（如从8减到4）
   - 使用 `--dtype float32` 而不是 `float64`
   - 减小 `--max-radius`（减少邻居数量）

3. **数据格式**: 确保XYZ文件格式正确，特别是能量、力和应力的单位。能量以 eV 为单位，力以 eV/Å 为单位，应力以 eV/Å³ 为单位。

4. **检查点**: 训练过程中会定期保存检查点（频率由`--dump-frequency`控制），建议定期备份。最终模型会保存在`--checkpoint`指定的路径。

5. **日志**: 训练日志保存在 `training_*.log` 文件中，包含详细的训练信息。控制台只显示验证结果和重要信息，详细日志请查看日志文件。验证阶段每个batch的能量预测默认只记录到日志文件，如需在控制台显示，请使用 `--log-val-batch-energy` 参数。

6. **早停**: 如果**加权验证损失**（`a × 能量损失 + b × 受力损失 + c × 应力损失`）连续`--patience`个epoch没有改善，训练会自动停止。这可以防止过拟合并节省时间。早停依据与训练时的总损失保持一致。

7. **权重自动调整**: `a`和`b`会在训练过程中自动调整（每`--update-param`个batch）。`a`会乘以`--weight-a-growth`（默认1.05，即增长5%），`b`会乘以`--weight-b-decay`（默认0.98，即减少2%），以平衡能量和受力的重要性。若启用应力训练（`-c > 0`），`c`参与总损失计算。如果训练不稳定，可以：
   - 增大`--update-param`（减少调整频率）
   - 使用更慢的调整速率（`--weight-a-growth 1.005 --weight-b-decay 0.995`）
   - 固定权重（设置`--update-param`为一个很大的值，如1000000）

8. **学习率策略**: 学习率会经历三个阶段：
   - **预热阶段**（前`--warmup-batches`个batch）：从`learning_rate × warmup_start_ratio`线性增长到`learning_rate`
   - **稳定阶段**：保持`learning_rate`
   - **衰减阶段**：如果验证指标不改善，每`--lr-decay-patience`个batch后乘以`--lr-decay-factor`

## 张量积模式对比

**FusedSCEquiTensorPot 支持八种等变张量积实现模式**，每种模式在速度、参数量和等变性方面有不同的特点：

1. **`spherical`**: 基于 e3nn 的球谐函数方法（默认，标准实现）
2. **`spherical-save`**: channelwise edge conv（e3nn 后端，参数量更少）
3. **`spherical-save-cue`**: channelwise edge conv（cuEquivariance 后端，需可选依赖，GPU 加速）
4. **`partial-cartesian`**: 笛卡尔坐标 + e3nn CG 系数（严格等变）
5. **`partial-cartesian-loose`**: 近似等变（norm product 近似）
6. **`pure-cartesian`**: 纯笛卡尔 \(3^L\) 表示（严格等变，速度极慢，不推荐）
7. **`pure-cartesian-sparse`**: 稀疏纯笛卡尔（严格等变，参数量优化）
8. **`pure-cartesian-ictd`**: ICTD irreps 内部表示（严格等变，速度最快，参数量最少）

所有模式都保持 O(3) 等变性（包括旋转和反射）。以下是对比数据：

### 模式对比总览

| 特性 | Spherical | Partial-Cartesian | Partial-Cartesian-Loose | Pure-Cartesian | Pure-Cartesian-Sparse | Pure-Cartesian-ICTD |
|------|-----------|-------------------|------------------------|----------------|----------------------|---------------------|
| 实现 | e3nn 球谐函数 | 笛卡尔坐标 + e3nn CG 系数 | 非严格等变（norm product 近似，使用 e3nn Irreps） | 纯笛卡尔（3^L，δ/ε，完全自实现） | 稀疏纯笛卡尔（δ/ε，完全自实现） | ICTD irreps 内部表示（完全自实现） |
| 等变性 | ✅ 严格等变 | ✅ 严格等变 | ⚠️ 近似等变 | ✅ 严格等变 | ✅ 严格等变 | ✅ 严格等变 |
| 速度 (CPU, lmax=2) | 1.00x (基准) | 1.06x | 1.33x | 0.06x (极慢) | 1.39x | **4.12x (最快)** |
| 速度 (GPU, lmax=2)* | 1.00x (基准) | 0.75x | 1.15x | 0.06x (极慢) | 1.17x | **2.10x (最快)** |
| 参数量 (lmax=2) | 6,540,634 (100%) | 5,404,938 (82.6%) | 5,406,026 (82.7%) | 33,626,186 (514.0%) | 4,606,026 (70.4%) | 1,824,497 (27.9%) |
| 参数量变化 | - | **减少 17.4%** | **减少 17.3%** | +414% | **减少 29.6%** | **减少 72.1%** |
| 等变性误差 (O(3), lmax=2) | ~1e-15 | ~1e-14 | ~1e-15 | ~1e-14 | ~1e-15 | ~1e-7 |
| 推荐场景 | 默认，最高精度 | 严格等变，平衡性能 | 快速迭代（CPU），非严格等变可接受 | 不推荐（速度慢） | 参数量优化，严格等变 | **参数量最少，GPU 最快，严格等变** |

*GPU 速度：总训练时间（前向+反向）加速比，相对于 spherical。测试环境：RTX 3090, float64, N=32, E=256。

### 使用方法

**训练时选择模式：**
```bash
# 球谐模式（默认，精度最高）
mff-train --input-file data.xyz --data-dir output

# 笛卡尔模式（速度快，参数少）
mff-train --input-file data.xyz --data-dir output --tensor-product-mode partial-cartesian
```

**评估时必须使用相同模式：**
```bash
# 如果训练时用的是 cartesian
mff-evaluate --checkpoint model.pth --tensor-product-mode partial-cartesian
```

### 详细性能对比

#### CPU 测试结果（channels=64, lmax=0 到 6, 32 atoms, 256 edges, float64）

**总训练时间加速比（前向+反向，相对于 spherical）：**

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**参数量对比（lmax=0 到 6）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**等变误差对比（O(3)，包含宇称）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 3.33e-15  |        8.24e-16  |            6.39e-14  |      3.41e-15  |           4.34e-15  |           1.69e-12 |
| 2    | 3.88e-15  |        1.87e-14  |            7.73e-16  |      3.08e-15  |           3.56e-14  |           7.73e-08 |
| 4    | 1.27e-15  |        6.83e-15  |            1.00e-15  |       **FAILED** |           1.24e-14  |           3.50e-05 |
| 6    | 3.26e-15  |        2.01e-15  |            5.82e-16  |       **FAILED** |           1.51e-15  |           1.00e-06 |

#### GPU 测试结果（channels=64, lmax=0 到 6, RTX 3090, float64, N=32, E=256）

**总训练时间加速比（前向+反向，相对于 spherical）：**

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**参数量对比（lmax=0 到 6）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**等变误差对比（O(3)，包含宇称）：**

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 1.03e-15  |        6.37e-16  |            1.20e-15  |      3.27e-15  |           9.31e-16  |           5.24e-08 |
| 2    | 9.27e-16  |        1.97e-16  |            8.96e-16  |      3.18e-14  |           1.30e-15  |           7.18e-07 |
| 4    | 9.16e-16  |        7.76e-14  |            4.26e-16  |       **FAILED** |           6.19e-16  |           7.16e-07 |
| 6    | 4.77e-16  |        7.02e-16  |            5.65e-16  |       **FAILED** |           5.72e-16  |           1.11e-07 |

**性能分析：**
- ✅ **所有模式均通过 O(3) 等变性测试**（包括宇称/反射），等变性误差 < 1e-6
- 🚀 **CPU 上 `pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **4.12x 加速** at lmax=2）
- 🚀 **GPU 上 `pure-cartesian-ictd` 在所有 lmax 下都是最快的**（最高 **2.10x 加速** at lmax=2）
- 🚀 **CPU/GPU 上 `pure-cartesian-sparse` 表现稳定**（0.53x - 1.39x，接近基准）
- 💾 **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- 💾 **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- ⚠️ **`pure-cartesian` 极慢且 lmax≥4 时失败**（不推荐）
- 📊 **CPU 测试环境**：channels=64, lmax=0-6, 32 atoms, 256 edges, float64
- 📊 **GPU 测试环境**：channels=64, lmax=0-6, RTX 3090, float64, 32 atoms, 256 edges

### 推荐使用场景

**使用 Spherical（默认）：**
- ✅ 需要最高精度和兼容性
- ✅ 研究/发表论文场景（标准 e3nn 实现）
- ✅ 首次尝试/对比基准
- ✅ 小规模数据集

**使用 Partial-Cartesian：**
- ✅ 需要严格等变性但参数量更少（减少 17.4%）
- ✅ GPU 内存受限场景
- ✅ 需要平衡性能和等变性

**使用 Partial-Cartesian-Loose：**
- ✅ 快速实验迭代（速度较快，CPU: 0.17x-1.37x，GPU: 0.21x-1.52x）
- ⚠️ 对严格等变性要求不高的场景（使用 norm product 近似，非严格等变）
- ⚠️ 注意：虽然等变性误差 < 1e-6，但理论上非严格等变

**使用 Pure-Cartesian-Sparse（推荐用于 CPU/GPU 训练）：**
- ✅ **CPU 上表现稳定**（0.53x - 1.39x，接近基准）
- ✅ **GPU 上速度较快**（**1.17x 加速** at lmax=2）
- ✅ 参数量适中（减少 29.6%，70-88% of spherical）
- ✅ 严格等变（误差 ~1e-15，最高精度）
- ✅ 平衡参数量和性能的最佳选择

**使用 Pure-Cartesian-ICTD（推荐用于 CPU/GPU 训练）：**
- ✅ **CPU 上速度最快**（最高 **4.12x 加速** at lmax=2，所有 lmax 下 1.58x - 4.12x）
- ✅ **GPU 上速度最快**（最高 **2.10x 加速** at lmax=2，所有 lmax 下 1.05x - 2.10x）
- ✅ 参数量最少（减少 72.1%，27-32% of spherical）
- ✅ 严格等变（误差 ~1e-7 到 1e-6，可接受）
- ✅ 内存极度受限场景
- ✅ 大规模模型部署
- 📊 **CPU 最佳性能**：lmax ≤ 3 时优势最明显（2.68x - 4.12x 加速）
- 📊 **GPU 最佳性能**：lmax ≤ 3 时优势最明显（1.91x - 2.10x 加速）

### 实际任务测试结果

**数据集**：五条氮氧化物和碳结构反应路径的 NEB（Nudged Elastic Band）数据，截取到 fmax=0.2，总共 2,788 条数据。测试集：每个反应选取 1-2 条完整或不完整的数据。

**测试配置**：64 channels, lmax=2, float64

<table>
<thead>
<tr>
<th style="text-align:center">方法</th>
<th style="text-align:center">配置</th>
<th style="text-align:center">模式</th>
<th style="text-align:center">能量 RMSE<br/>(mev/atom)</th>
<th style="text-align:center">力 RMSE<br/>(mev/Å)</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="3" style="text-align:center;vertical-align:middle"><strong>MACE</strong></td>
<td style="text-align:center">Lmax=2, 64ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.13</td>
<td style="text-align:center">11.6</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 128ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.12</td>
<td style="text-align:center">11.3</td>
</tr>
<tr>
<td style="text-align:center">Lmax=2, 198ch</td>
<td style="text-align:center">-</td>
<td style="text-align:center">0.24</td>
<td style="text-align:center">15.1</td>
</tr>
<tr>
<td rowspan="4" style="text-align:center;vertical-align:middle"><strong>FSCETP</strong></td>
<td rowspan="4" style="text-align:center;vertical-align:middle">Lmax=2, 64ch</td>
<td style="text-align:center"><strong>spherical</strong></td>
<td style="text-align:center"><strong>0.044</strong> ⭐</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center"><strong>partial-cartesian</strong></td>
<td style="text-align:center">0.045</td>
<td style="text-align:center"><strong>7.4</strong> ⭐</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian-loose</td>
<td style="text-align:center">0.048</td>
<td style="text-align:center">8.4</td>
</tr>
<tr>
<td style="text-align:center">pure-cartesian-ictd</td>
<td style="text-align:center">0.046</td>
<td style="text-align:center">9.0</td>
</tr>
</tbody>
</table>

**结果分析**：
- **能量精度对比**：FSCETP 相比 MACE (64ch) 的能量 RMSE 降低了 66.2%（0.044 vs 0.13 mev/atom）
- **力精度对比**：FSCETP 相比 MACE (64ch) 的力 RMSE 降低了 36.2%（7.4 vs 11.6 mev/Å）
- **最佳性能模式**：`spherical` 和 `partial-cartesian` 模式达到最优精度（能量：0.044-0.045 mev/atom，力：7.4 mev/Å）
- **精度与效率平衡**：`pure-cartesian-ictd` 在保持接近最优精度（能量：0.046 mev/atom，力：9.0 mev/Å）的同时，参数量减少 72.1%，训练速度提升 2.10x（GPU，lmax=2）

**不推荐使用 Pure-Cartesian：**
- ❌ 速度极慢（CPU: 0.02x-0.36x，GPU: 0.02x-0.54x），参数量最大（+414%）
- ❌ lmax≥4 时失败（内存不足）
- ❌ 仅用于研究目的，不推荐实际使用

### 完整示例：笛卡尔模式 + 多卡 + SWA + EMA

```bash
torchrun --nproc_per_node=4 -m molecular_force_field.cli.train \
    --input-file large_dataset.xyz \
    --data-dir data \
    --tensor-product-mode partial-cartesian \
    --distributed \
    --epochs 2000 \
    --batch-size 8 \
    --lmax 2 \
    --irreps-output-conv-channels 64 \
    -a 1.0 -b 10.0 \
    --swa-start-epoch 1000 --swa-a 100 --swa-b 10 \
    --ema-start-epoch 1500 --ema-decay 0.999 --use-ema-for-validation --save-ema-model \
    --patience 50 \
    --device cuda
```

### Python API 使用

```python
from molecular_force_field.models import CartesianTransformerLayer
from molecular_force_field.utils.config import ModelConfig

config = ModelConfig(
    max_atomvalue=10,
    embedding_dim=16,
    lmax=2,
    channel_in=64,
)

# 创建笛卡尔模型
model = CartesianTransformerLayer(
    max_embed_radius=config.max_radius,
    main_max_radius=config.max_radius_main,
    main_number_of_basis=config.number_of_basis_main,
    hidden_dim_conv=config.channel_in,
    hidden_dim_sh=config.get_hidden_dim_sh(),
    hidden_dim=config.emb_number_main_2,
    channel_in2=config.channel_in2,
    embedding_dim=config.embedding_dim,
    max_atomvalue=config.max_atomvalue,
    output_size=config.output_size,
    embed_size=config.embed_size,
    main_hidden_sizes3=config.main_hidden_sizes3,
    num_layers=config.num_layers,
    function_type_main=config.function_type,
    lmax=config.lmax,
    device='cuda'
).to('cuda')

# 前向传播
energy = model(pos, Z, batch, edge_src, edge_dst, edge_shifts, cell)
forces = -torch.autograd.grad(energy.sum(), pos)[0]
```

### 笛卡尔张量积 API

如果你需要单独使用笛卡尔张量积层：

```python
from molecular_force_field.models import CartesianFullyConnectedTensorProduct

# 创建张量积（类似 e3nn.o3.FullyConnectedTensorProduct）
tp = CartesianFullyConnectedTensorProduct(
    irreps_in1="64x0e + 64x1o + 64x2e",
    irreps_in2="1x0e + 1x1o + 1x2e",
    irreps_out="64x0e + 64x1o + 64x2e",
    shared_weights=True,
    internal_weights=True
)

# 前向传播
out = tp(x1, x2)

# 或者使用外部权重
tp_ext = CartesianFullyConnectedTensorProduct(
    irreps_in1="64x0e + 64x1o + 64x2e",
    irreps_in2="1x0e + 1x1o + 1x2e",
    irreps_out="64x0e",
    shared_weights=False,
    internal_weights=False
)
weights = weight_network(edge_features)  # shape: (..., tp_ext.weight_numel)
out = tp_ext(x1, x2, weights)
```
