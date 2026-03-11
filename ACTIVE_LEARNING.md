# 主动学习 (mff-active-learn)

主动学习实现的工作流：**训练集成 → 探索（MD/NEB）→ 按力偏差筛选 → DFT 标注 → 合并数据 → 重复**，用于在势能面尚未覆盖的区域自动采点并扩充训练集。

支持 **单节点**（本机多进程并发标注）和 **超算**（SLURM 按结构提交作业）。同一套 DFT 脚本模板可同时用于本地测试（`local-script`）和集群（`slurm`）。

---

## 冷启动：从零生成初始数据集 (mff-init-data)

当只有一个或几个种子结构、没有已标注数据时，`mff-init-data` 可一键完成：**扰动 → DFT 标注 → 预处理**，输出可直接用于训练或 AL 的完整数据集。

```bash
# 分子体系（PySCF，无需外部二进制）
mff-init-data --structures water.xyz ethanol.xyz \
    --n-perturb 15 --rattle-std 0.05 \
    --label-type pyscf --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 4 --output-dir data

# 周期性体系（VASP，含晶胞缩放）
mff-init-data --structures POSCAR.vasp \
    --n-perturb 20 --rattle-std 0.02 --cell-scale-range 0.03 \
    --label-type vasp --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4 \
    --label-n-workers 8 --output-dir data

# 从目录自动收集所有种子结构
mff-init-data --structures structures/ \
    --n-perturb 10 --label-type pyscf --output-dir data
```

### 扰动参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--structures` | 必选 | 一个或多个种子结构文件，或包含 .xyz/.cif 文件的目录 |
| `--n-perturb` | 10 | 每个种子结构生成的扰动数量（不含原始结构本身） |
| `--rattle-std` | 0.05 | 原子位移高斯分布 σ (Å)。分子用 0.03–0.1，晶体用 0.01–0.03 |
| `--cell-scale-range` | 0 | 晶胞随机缩放 ±范围（仅周期性体系）。如 0.03 表示 ±3% |
| `--min-dist` | 0.5 | 最小原子间距过滤 (Å)，丢弃非物理构型 |
| `--train-ratio` | 0.9 | 训练/验证集比例 |
| `--skip-preprocess` | False | 仅输出标注后的 XYZ，跳过 H5 预处理 |

> **典型流程**：`mff-init-data` 生成初始数据集 → `mff-active-learn` 迭代扩充。

---

## 快速开始示例

### PySCF（本地，无需外部 DFT 二进制）

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 8 --md-steps 500 --n-iterations 5
```

### VASP（单节点，ASE 接口）

```bash
export ASE_VASP_COMMAND="mpirun -np 4 vasp_std"
export VASP_PP_PATH=/path/to/potpaw_PBE

mff-active-learn --explore-type ase --label-type vasp \
    --vasp-xc PBE --vasp-encut 500 --vasp-kpts 4 4 4 \
    --label-n-workers 8 --label-threads-per-worker 4 \
    --md-steps 1000 --n-iterations 10
```

### SLURM（超算，每结构一个作业）

```bash
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template dft_job.sh --slurm-partition cpu \
    --slurm-nodes 1 --slurm-ntasks 32 --slurm-time 04:00:00
```

### local-script（本地执行脚本模板）

```bash
mff-active-learn --explore-type ase --label-type local-script \
    --local-script-template dft_job.sh \
    --label-n-workers 4 \
    --md-steps 500 --n-iterations 3
```

---

## 核心参数表

| 参数 | 默认 | 说明 |
|------|------|------|
| `--work-dir` | `al_work` | 主动学习工作目录 |
| `--data-dir` | `data` | 训练数据目录（含 processed_train.h5 或 train.xyz） |
| `--init-structure` | 自动从 data-dir 提取 | 一个或多个初始结构路径，或包含 .xyz 文件的目录（多结构并行探索） |
| `--init-checkpoint` | 无 | 可选 warm start checkpoint。第 0 轮可跳过训练直接探索；1 个 checkpoint=bootstrap 模式，`n_models` 个 checkpoint=完整集成 |
| `--n-models` | 4 | 集成模型数量 |
| `--n-iterations` | 20 | 单阶段最大迭代次数（不用 stages 时） |
| `--explore-type` | 必选 | 探索后端：`ase` 或 `lammps` |
| `--explore-mode` | `md` | 探索方式：`md`（分子动力学）或 `neb`（弹性带） |
| `--explore-n-workers` | 1 | 多结构并行探索线程数；`1`=顺序，`>1`=并发（ThreadPoolExecutor） |
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
| `--train-n-gpu` | 1 | 每个集成模型训练使用的 GPU 数（每节点）。1=单卡/CPU，>1 自动用 torchrun DDP |
| `--train-max-parallel` | 0 (auto) | 同时训练的最大模型数。0=自动（可用GPU÷n_gpu），1=串行。多节点时强制为1 |
| `--train-nnodes` | 1 | 每个模型使用的节点数。1=单节点，>1=多节点 DDP |
| `--train-master-addr` | auto | rendezvous 地址（auto=从 SLURM 或本机 hostname 解析） |
| `--train-master-port` | 29500 | 基础端口（并行模型自动偏移避免冲突） |
| `--train-launcher` | auto | 启动器：auto / local / slurm（SLURM 下自动分配节点子集并行训练） |
| `--resume` | 关闭 | 从 `work_dir/al_state.json` 和已有 `iterations/iter_*` 产物恢复主动学习 |
| `--stages` | 无 | 多阶段 JSON 文件路径 |
| `--device` | `cuda` | 推理设备 |
| `--max-radius` | 5.0 | 邻居搜索最大半径 (Å) |
| `--atomic-energy-file` | `data/fitted_E0.csv` | 原子参考能量 CSV |
| `--neb-initial` / `--neb-final` | 无 | NEB 模式下的初/末结构 |

---

## 多层筛选

候选构型经过三层筛选后再送标注，显著降低 DFT 成本并提升训练集多样性。

| 层 | 名称 | 说明 |
|----|------|------|
| **Layer 0** | 失败帧恢复 | 可选地将部分 `fail` 帧（`max_devi_f ≥ level_f_hi`）中最不极端的构型纳入候选 |
| **Layer 1** | 不确定性门控 | 保留 `level_f_lo ≤ max_devi_f < level_f_hi` 的帧（DPGen2 信任窗口） |
| **Layer 2** | 多样性筛选 | 用结构指纹（SOAP / deviation 直方图）+ FPS 选取最大化多样性的子集 |

### 多样性筛选参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--diversity-metric` | `soap` | 指纹类型：`soap`（需 dscribe）、`devi_hist`（零额外推理）、`none`（不筛） |
| `--max-candidates-per-iter` | 50 | 每轮多样性筛选后最多保留的候选数 |
| `--soap-rcut` | 5.0 | SOAP 截断半径 (Å) |
| `--soap-nmax` | 8 | SOAP 径向基展开阶数 |
| `--soap-lmax` | 6 | SOAP 角向展开阶数 |
| `--soap-sigma` | 0.5 | SOAP 高斯展宽 |
| `--devi-hist-bins` | 32 | `devi_hist` 直方图桶数 |

### 失败帧处理参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--fail-strategy` | `discard` | `discard`（丢弃所有 fail 帧）或 `sample_topk`（取最温和的 fail 帧加入候选） |
| `--fail-max-select` | 10 | `sample_topk` 时最多纳入的 fail 帧数 |

> **依赖**：SOAP 指纹需要 `dscribe`。安装：`pip install dscribe` 或 `pip install molecular_force_field[al]`。
> 若未安装 dscribe，`--diversity-metric soap` 会自动回退为 `devi_hist`。

---

## 多结构并行探索

当训练集包含多种不同结构（如不同分子、不同晶体构型、不同组分）时，可传入多个初始结构，
每次迭代会分别从每个结构出发做 MD 探索，然后将所有轨迹合并后统一进行偏差计算和多层筛选。

### 用法

```bash
# 直接传入多个文件，顺序执行（默认）
mff-active-learn --init-structure struct_A.xyz struct_B.xyz struct_C.xyz \
    --explore-type ase --label-type pyscf ...

# 并行探索：同时跑 3 个 MD（每个结构占一个线程）
mff-active-learn --init-structure struct_A.xyz struct_B.xyz struct_C.xyz \
    --explore-n-workers 3 \
    --explore-type ase --label-type pyscf ...

# 传入一个目录（自动收集所有 .xyz / .cif 文件），并行 4 个线程
mff-active-learn --init-structure structures/ \
    --explore-n-workers 4 \
    --explore-type ase --label-type pyscf ...
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--explore-n-workers` | `1` | 多结构探索并行线程数；`1` = 顺序执行；`>1` 启用 `ThreadPoolExecutor` |

> **注意**：每个线程会独立加载模型并占用显存/内存。GPU 机器上建议
> `explore_n_workers × 每个 MD 的显存 ≤ 总显存`，否则请用 CPU 做 MD
> 或减小 `--explore-n-workers`。

### 工作流程

每次迭代中：

1. **并行探索**：`--explore-n-workers` 个线程同时从各自初始结构出发运行 MD，
   生成独立子轨迹（`explore_traj_0.xyz`, `explore_traj_1.xyz`, ...）
2. **轨迹合并**：合并所有子轨迹为 `explore_traj.xyz`
3. **统一筛选**：对合并轨迹做模型偏差计算 → Layer 0/1 不确定性门控 → Layer 2 多样性筛选
4. **标注与合并**：筛选后的候选帧统一标注并合入训练集

多样性筛选（SOAP / devi_hist）会自动平衡来自不同结构的构型，确保训练集覆盖所有系统类型。

---

## 标注类型 (--label-type)

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

---

## 并发与线程控制

| 参数 | 默认 | 说明 |
|------|------|------|
| `--label-n-workers` | 1 | 同时跑多少个结构（进程数） |
| `--label-threads-per-worker` | 1（n_workers>1 时）或自动 | 每个结构内部线程数（如 PySCF/VASP 的 OpenMP） |
| `--label-error-handling` | `raise` | `raise`（任一失败即退出）或 `skip`（跳过失败结构继续） |

**建议**：`n_workers × threads_per_worker ≤ 总核数`，避免过载。

---

## 多阶段 JSON 格式 (--stages)

通过 JSON 定义多个阶段，每阶段可设不同温度、步数、迭代上限和收敛阈值。例如先 300K 再 600K：

```bash
mff-active-learn --explore-type ase --label-type pyscf --stages stages.json
```

`stages.json` 示例（数组，每项一个阶段）：

```json
[
  {
    "name": "300K",
    "temperature": 300,
    "nsteps": 500,
    "timestep": 1.0,
    "log_interval": 10,
    "level_f_lo": 0.05,
    "level_f_hi": 0.5,
    "conv_accuracy": 0.9,
    "max_iters": 5
  },
  {
    "name": "600K",
    "temperature": 600,
    "nsteps": 1000,
    "timestep": 1.0,
    "level_f_lo": 0.05,
    "level_f_hi": 0.5,
    "conv_accuracy": 0.9,
    "max_iters": 5
  }
]
```

也支持 `{"stages": [...]}` 包裹格式。未给出的字段会使用默认值。使用 `--stages` 时，命令行中的 `--md-*`、`--n-iterations` 等单阶段参数会被忽略。

---

## 脚本模板占位符

`local-script` 与 `slurm` 共用同一套占位符：

| 占位符 | 说明 |
|--------|------|
| `{run_dir}` | 当前结构的运行目录 |
| `{input_xyz}` | 输入 XYZ 路径 |
| `{output_xyz}` | 输出 extended XYZ 路径 |
| `{job_name}` | 作业/任务名称 |
| `{partition}` | SLURM 分区（仅 slurm） |
| `{nodes}` | 节点数（仅 slurm） |
| `{ntasks}` | 任务数（仅 slurm） |
| `{time}` | 时间限制（仅 slurm） |
| `{mem}` | 内存（仅 slurm） |

模板内其他 `{key}` 或 shell 变量（如 `$HOME`）会原样保留。

---

## SLURM 参数

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

若某结构的 `output.xyz` 已存在（例如上次中断后重跑），会自动跳过该结构的提交（resume）。

---

## DFT 后端 CLI 参数

### PySCF

| 参数 | 默认 | 说明 |
|------|------|------|
| `--pyscf-method` | `b3lyp` | 方法：b3lyp, pbe, hf, mp2 等 |
| `--pyscf-basis` | `sto-3g` | 基组：sto-3g, 6-31g*, def2-svp 等 |
| `--pyscf-charge` | 0 | 总电荷 |
| `--pyscf-spin` | 0 | 2S（未配对电子数） |
| `--pyscf-max-memory` | 4000 | 最大内存 (MB) |
| `--pyscf-conv-tol` | 1e-9 | SCF 收敛阈值 |

### VASP

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

### CP2K

| 参数 | 默认 | 说明 |
|------|------|------|
| `--cp2k-xc` | `PBE` | XC 泛函 |
| `--cp2k-basis-set` | `DZVP-MOLOPT-SR-GTH` | 高斯基组 |
| `--cp2k-pseudo` | `auto` | 赝势名称 |
| `--cp2k-cutoff` | 400.0 | 平面波截断 (Ry) |
| `--cp2k-max-scf` | 50 | 最大 SCF 迭代 |
| `--cp2k-charge` | 0.0 | 总电荷 |
| `--cp2k-command` | 覆盖 ASE_CP2K_COMMAND | 运行命令 |
| `--cp2k-cleanup` | False | 成功后删除运行目录 |

### Quantum Espresso

| 参数 | 默认 | 说明 |
|------|------|------|
| `--qe-pseudo-dir` | 必选 | 赝势 .UPF 目录 |
| `--qe-pseudopotentials` | 必选 | JSON：`'{"H":"H.pbe.UPF","O":"O.pbe.UPF"}'` |
| `--qe-ecutwfc` | 60.0 | 波函数截断 (Ry) |
| `--qe-ecutrho` | 4*ecutwfc | 电荷密度截断 (Ry) |
| `--qe-kpts` | `1 1 1` | k 点网格 |
| `--qe-command` | 无 | pw.x 命令 |
| `--qe-cleanup` | False | 成功后删除运行目录 |

### Gaussian

| 参数 | 默认 | 说明 |
|------|------|------|
| `--gaussian-method` | `b3lyp` | 理论级别 |
| `--gaussian-basis` | `6-31+G*` | 基组 |
| `--gaussian-charge` | 0 | 总电荷 |
| `--gaussian-mult` | 1 | 自旋多重度 2S+1 |
| `--gaussian-nproc` | 1 | %nprocshared |
| `--gaussian-mem` | `4GB` | 内存 |
| `--gaussian-command` | 无 | 覆盖 Gaussian 命令 |
| `--gaussian-cleanup` | False | 成功后删除运行目录 |

### ORCA

| 参数 | 默认 | 说明 |
|------|------|------|
| `--orca-simpleinput` | `B3LYP def2-TZVP TightSCF` | 简单输入行（! 之后） |
| `--orca-nproc` | 1 | %pal nprocs |
| `--orca-charge` | 0 | 总电荷 |
| `--orca-mult` | 1 | 自旋多重度 2S+1 |
| `--orca-command` | 无 | ORCA 可执行路径 |
| `--orca-cleanup` | False | 成功后删除运行目录 |

---

## 环境变量

| 变量 | 说明 |
|------|------|
| `ASE_VASP_COMMAND` | VASP 运行命令，如 `mpirun -np 4 vasp_std` |
| `VASP_PP_PATH` | VASP 赝势目录 |
| `ASE_CP2K_COMMAND` | CP2K 运行命令，如 `cp2k_shell.psmp` |
| `CP2K_DATA_DIR` | CP2K 数据目录 |
| `ASE_GAUSSIAN_COMMAND` | Gaussian 命令，如 `g16 < PREFIX.com > PREFIX.log` |

ORCA 和 QE 可通过 `--orca-command`、`--qe-command` 指定，或确保可执行文件在 PATH 中。

---

## 使用示例汇总

**快速测试（不跑 DFT，用 ML 自举）：**

```bash
mff-active-learn --explore-type ase --explore-mode md --label-type identity \
    --md-temperature 300 --md-steps 200 --n-iterations 2 --epochs 5 \
    --n-models 2
```

**本地 PySCF + 多 worker：**

```bash
mff-active-learn --explore-type ase --label-type pyscf \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --label-n-workers 4 --md-steps 500 --n-iterations 5 --epochs 100
```

**超算 SLURM + 脚本模板：**

```bash
mff-active-learn --explore-type ase --label-type slurm \
    --slurm-template vasp_job.sh \
    --slurm-partition normal --slurm-ntasks 64 --slurm-time 08:00:00 \
    --stages stages.json --label-error-handling skip
```

**多结构并行探索（不同构型/分子共同学习，并发 3 线程）：**

```bash
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure mol_A.xyz mol_B.xyz mol_C.xyz \
    --explore-n-workers 3 \
    --pyscf-method b3lyp --pyscf-basis 6-31g* \
    --diversity-metric soap --max-candidates-per-iter 50 \
    --md-steps 1000 --n-iterations 10
```

**从已有 checkpoint 直接开始第 0 轮探索（跳过首轮训练）：**

```bash
# 单 checkpoint：bootstrap 模式
# 第 0 轮直接 MD -> 候选/多样性 -> 标注，不走 uncertainty gate
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --init-checkpoint warm_start.pth \
    --n-models 4 \
    --md-steps 1000 --n-iterations 10

# 完整集成 warm start：提供 n_models 个 checkpoint
# 第 0 轮仍跳过训练，但可直接计算 ensemble deviation
mff-active-learn --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --init-checkpoint model_0.pth model_1.pth model_2.pth model_3.pth \
    --n-models 4 \
    --md-steps 1000 --n-iterations 10
```

**主动学习中断后恢复：**

```bash
mff-active-learn --work-dir al_work --data-dir data \
    --explore-type ase --label-type pyscf \
    --init-structure seed.xyz \
    --resume
```

恢复时会自动读取 `al_state.json`，并尽量复用已有的：

1. `checkpoint/` 训练结果
2. `explore_traj.xyz`
3. `model_devi.out`
4. `candidate.xyz`
5. `labeled.xyz`
6. `merge.done`

**NEB 探索：**

```bash
mff-active-learn --explore-type ase --explore-mode neb \
    --neb-initial reactant.xyz --neb-final product.xyz \
    --label-type pyscf --pyscf-method b3lyp --n-iterations 5
```

---

## FAQ

### Q: 初始结构从哪里来？

若未指定 `--init-structure`，会从 `--data-dir` 的 `train.xyz` 或 `processed_train.h5` 取第一个结构。
支持传入多个文件或一个目录实现**多结构并行探索**——详见上方「多结构并行探索」一节。

### Q: 如何从上次中断处继续？

直接加 `--resume` 重新运行即可。主动学习会读取 `work_dir/al_state.json`，并自动复用当前 `iterations/iter_*` 下已存在的训练 checkpoint、探索轨迹、`model_devi.out`、候选集、标注结果和 `merge.done` 标记。

这意味着：

1. 若上次中断在训练后、探索前，会复用已训练好的 checkpoint
2. 若中断在探索后、标注前，会复用轨迹和 `model_devi.out`
3. 若中断在 merge 过程中，成功 merge 后会写 `merge.done`，下次 resume 不会重复 merge

对于 `slurm` 标注器，其内部也仍然会跳过已有 `output.xyz` 的结构任务。

### Q: 只有一个初始 checkpoint，也能直接开始主动学习吗？

可以。传 `--init-checkpoint warm_start.pth` 后，第 0 轮会进入 **bootstrap 模式**：

1. 跳过训练
2. 直接用该 checkpoint 跑 MD 探索
3. 将探索帧直接送入候选池，再做可选多样性筛选和标注

由于单 checkpoint 无法计算 ensemble deviation，第 0 轮不会走 Layer 0/1 uncertainty gate。
从第 1 轮开始会恢复正常的“训练集成 -> deviation -> 筛样”流程。

### Q: 输出 extended XYZ 格式要求？

标注器输出的 XYZ 需包含 `Properties=species:S:1:pos:R:3:energy:R:1:forces:R:3` 及相应数据，能量单位 eV，力单位 eV/Å。

### Q: 如何查看所有参数？

```bash
mff-active-learn --help
```

### Q: 训练超参数如何传递？

`--epochs` 会传给内部 `mff-train`。其他训练参数可通过扩展 `train_args` 传入（当前 CLI 仅暴露 `--epochs`）。

---

## ASE Calculator 进阶用法

`MyE3NNCalculator` 支持当前实现中的全部 `tensor_product_mode`，还提供外场和物理张量输出的扩展接口。

### 1. 所有张量积模式

以下模式均可通过 `build_e3trans_from_checkpoint` 加载后传给 `MyE3NNCalculator`：

| 模式 | 说明 |
|------|------|
| `spherical` | 标准球谐 TP |
| `spherical-save` | 节省显存版球谐 TP |
| `spherical-save-cue` | CUE 算子加速版球谐 TP（新增） |
| `partial-cartesian` | 部分笛卡尔 TP |
| `partial-cartesian-loose` | 宽松版部分笛卡尔 TP |
| `pure-cartesian` | 纯笛卡尔 TP |
| `pure-cartesian-sparse` | 稀疏纯笛卡尔 TP |
| `pure-cartesian-ictd` | ICTD 纯笛卡尔 TP（支持外场和物理张量） |
| `pure-cartesian-ictd-save` | 节省显存版 ICTD |

### 2. 外场支持（ICTD 模型）

若模型训练时启用了 `--external-tensor-rank`（如电场、磁场），需在 Calculator 中传入 `external_tensor`：

```python
import torch
from molecular_force_field.evaluation.calculator import MyE3NNCalculator
from molecular_force_field.active_learning.model_loader import build_e3trans_from_checkpoint

device = torch.device("cuda")
model, config = build_e3trans_from_checkpoint("checkpoint.pt", device)
ref_dict = dict(zip(
    [k.item() for k in config.atomic_energy_keys],
    [v.item() for v in config.atomic_energy_values],
))

# 电场向量 (3,)，单位 V/Å
E_field = torch.tensor([0.0, 0.0, 0.01], device=device, dtype=torch.float64)

calc = MyE3NNCalculator(
    model, ref_dict, device, max_radius=5.0,
    external_tensor=E_field,
)
atoms.calc = calc
energy = atoms.get_potential_energy()  # 含电场项
```

`external_tensor=None`（默认）表示无外场，与非 ICTD 模型完全兼容。

### 3. 物理张量输出（如偶极矩、极化率）

ICTD 模型可选择性地输出多极矩或极化率等物理张量：

```python
calc = MyE3NNCalculator(
    model, ref_dict, device, max_radius=5.0,
    return_physical_tensors=True,
)
atoms.calc = calc
atoms.get_potential_energy()

# 结果存储在 results["physical_tensors"]：
# 格式：{名称: {l阶: np.ndarray(...)}}
pt = calc.results["physical_tensors"]
dipole = pt["dipole"][1]      # l=1 张量（3 个分量）
polar  = pt["polarizability"][2]  # l=2 张量（5 个分量）
```

> **注意**：只有使用 `--physical-tensors` 训练的 ICTD 检查点才有物理张量输出头。
> 普通模型传入 `return_physical_tensors=True` 会引发错误。


