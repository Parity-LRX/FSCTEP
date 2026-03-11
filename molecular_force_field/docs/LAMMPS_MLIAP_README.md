# 将本框架以 ML-IAP unified 方式接入 LAMMPS

## 0. 模型限制

**仅以下五种模型支持 ML-IAP**，因其 forward 支持 `precomputed_edge_vec`：

| 模型文件 | tensor_product_mode |
|----------|---------------------|
| `e3nn_layers.py` | spherical |
| `e3nn_layers_channelwise.py` | spherical-save |
| `cue_layers_channelwise.py` | **spherical-save-cue**（cuEquivariance GPU 加速） |
| `pure_cartesian_ictd_layers.py` | pure-cartesian-ictd-save |
| `pure_cartesian_ictd_layers_full.py` | pure-cartesian-ictd |

其他模型（如 pure-cartesian、pure-cartesian-sparse、partial-cartesian 等）暂不支持 ML-IAP 导出。

### spherical-save-cue 的 ML-IAP 说明

- **依赖**：需安装 cuEquivariance（`pip install cuequivariance-torch cuequivariance-ops-torch-cu12`）
- **导出**：`python -m molecular_force_field.cli.export_mliap checkpoint.pth --elements H O --tensor-product-mode spherical-save-cue`
- **序列化限制**：cuEquivariance 内部模块可能无法 pickle，若 `torch.save` 失败，ML-IAP 无法使用该模型；可改用 `pure-cartesian-ictd` 或 `spherical-save` 作为替代

## 1. 模型接口

基于 `PureCartesianICTDTransformerLayer` 与 `inference_ddp.py` 的阅读：

| 项目 | 内容 |
|------|------|
| **前向输入** | `pos` (N,3), `A` (N,) 原子序数, `batch` (N,), `edge_src` (E,), `edge_dst` (E,), `edge_shifts` (E,3), `cell` (1,3,3) 或 (N,3,3) |
| **几何约定** | 边向量 `edge_vec = pos[edge_dst] - pos[edge_src] + shift_vecs`，其中 `shift_vecs = edge_shifts @ cell[batch[edge_src]]`（行向量乘 3x3 cell） |
| **前向输出** | `out` 形状 (N,)，即**每个节点的标量能量**；总能量 = `out.sum()` |
| **力** | 通过 `AtomForcesWrapper` 的 dummy `pos` autograd 得到 per-atom forces |
| **cutoff** | `main_max_radius = 5.0`（与 `max_embed_radius` 一致） |
| **消息层数** | `num_interaction = 2` |
| **PBC** | 支持；依赖 `cell` + `edge_shifts` 整数向量 |

## 2. LAMMPS ML-IAP unified data 接口

- **unified 对象**需实现：`compute_forces(data)`、`compute_descriptors(data)`、`compute_gradients(data)`（后两个可 `pass`）。
- **`data`** 里你能用的主要有：
  - `data.nlocal`, `data.ntotal`, `data.npairs`
  - `data.pair_i`、`data.jatoms`（或 `pair_j`）：每条边的 i、j 原子索引
  - `data.rij`：形状 `(npairs, 3)`，约定为 r_j - r_i
  - `data.elems`：每个原子的元素/类型 id
  - `data.f`：LAMMPS force buffer（numpy view / GPU tensor），可直接 `+=` 写入 per-atom forces
  - `data.energy`：总能量（直接赋值）
  - `data.eatoms`：per-atom 能量数组（直接赋值）
- **全局 virial**：LAMMPS C++ 端在 `compute_forces` 返回后自动调用 `virial_fdotr_compute()` 计算。

## 3. 当前实现：Atom Forces 模式

`LAMMPS_MLIAP_MFF` 使用 `AtomForcesWrapper`，通过 `dE/d(pos)`（per-atom）计算力，绕过了 per-pair edge forces（`dE/d(edge_vec)`）的 O(npairs) 梯度存储：

1. 创建 dummy `pos = zeros(ntotal, 3, requires_grad=True)` 作为 autograd 叶变量
2. 构建 `edge_vec = pos[dst] - pos[src] + rij.detach()`（rij 来自 LAMMPS）
3. 模型通过 `precomputed_edge_vec=edge_vec` 计算 per-atom 能量
4. `dE/d(pos)` → per-atom forces (N x 3)，直接写入 `data.f`
5. 设置 `data.energy` 和 `data.eatoms`
6. 全局 virial 由 LAMMPS 的 `virial_fdotr_compute()` 自动处理

**优势**：autograd 叶梯度从 O(npairs) 降至 O(natoms)，减小计算图和内存占用。

旧的 `EdgeForcesWrapper`（dE/d(edge_vec) -> per-pair forces -> `update_pair_forces`）保留在代码中作为 fallback，但不再是默认路径。

## 4. LAMMPS 侧需要和你一致的点

- **邻居列表 cutoff**：至少 `main_max_radius`（5.0 A）；若用 `ghostneigh_flag=1`，建议 `comm_modify cutoff` 设为 `num_interaction * main_max_radius + skin`（例如 `2*5.0+1.0=11.0`），否则深层消息传递会缺邻居。
- **元素/类型**：`pair_coeff * *` 后的元素列表要和训练时的 `A`（原子序数）对应；wrapper 里 `data.elems` 会被映射成原子序数。
- **cell**：使用 `data.rij` 时已是最近镜像向量，不需要再乘 cell。

## 5. 使用流程

1. **导出**：`python -m molecular_force_field.cli.export_mliap checkpoint.pth --elements H O`
2. **LAMMPS 输入**：`neighbor 1.0 bin`（减小 ghost 膨胀）+ `pair_style mliap unified model-mliap.pt 0` + `pair_coeff * * H O`
3. **LAMMPS 编译**：需 `PKG_ML-IAP=ON`、`MLIAP_ENABLE_PYTHON=ON`、`PKG_PYTHON=ON`、`BUILD_SHARED_LIBS=ON`；若用 Kokkos GPU，还需 `make install-python` 后从 Python 里 `lammps.mliap.activate_mliappy_kokkos(lmp)`。

## 6. 多卡 / 域分解

- 在 LAMMPS 里多 MPI rank 时，每个 rank 只算本域原子 + ghost 的邻居。
- 若用 **Kokkos GPU**，LAMMPS 会按域分解把子图交给 `compute_forces`，此时 `data` 里已是该 rank 的 local+ghost 原子和对应的 `npairs`、`rij`；只需在单进程内用模型前向 + atom forces 写入即可，不需要在 Python 里做 MPI 或 DDP 同步。
- 若希望"单一大体系、多 GPU 分摊"的图并行，需在 LAMMPS 外自己用 PyTorch DDP + 图分区跑；ML-IAP 这条是 LAMMPS 的域分解并行。
