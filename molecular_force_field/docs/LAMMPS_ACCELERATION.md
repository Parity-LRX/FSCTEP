# LAMMPS 加速方案：如何达到 ASE 级别的速度

## 现状对比

| 接口 | 50 原子 × 10 步 MD | 单步耗时 |
|------|-------------------|----------|
| **ASE Calculator** | ~0.5 s | ~17 ms/步 |
| **LAMMPS fix external** | ~7 s | ~200 ms/步 |

LAMMPS 约慢 **10–15 倍**，主要瓶颈在 **fix external pf/callback** 的 Python 回调机制。

---

## ML-IAP 性能“三大刺客”（实测 2000 原子体系）

| 刺客 | 现象 | 缓解措施 |
|------|------|----------|
| **1. Ghost 原子膨胀** | Nlocal=2000 → Ntotal=6839（含 4839 ghost），ntotal/nlocal≈3.4x 算力惩罚 | `neighbor 1.0 bin` 减小 skin（默认 metal 2.0 Å） |
| **2. CPU-GPU 数据搬运** | 100% CPU 使用 + 每步 `E_total.item()` 强制同步 | 确保 Kokkos 路径下无 `.cpu().numpy()`；`data.rij` 等由 LAMMPS 直接传 GPU 时避免拷贝 |
| **3. Neighbor Skin 过大** | ave neighs/atom≈137，MLIP 下偏大 | 同上 `neighbor 1.0 bin` |

**LAMMPS input 中务必加入**（在 `pair_style` 之前）：

```lammps
neighbor 1.0 bin
```

---

## 瓶颈分析

1. **C→Python 回调开销**：每步 LAMMPS C++ 调用 Python 回调，存在进程内切换和 GIL 开销。
2. **fix external 设计**：为通用外部势设计，非专为 ML 势优化。
3. **数据传递**：即使使用 `lmp.numpy.extract_atom` 减少复制，回调本身的调用成本仍占主导。

---

## 加速方案（按推荐顺序）

### 方案 1：用 ASE 跑 MD（推荐，零改动）

**适用**：只需做 MD、结构优化、NEB 等，不依赖 LAMMPS 特有功能。

```bash
python -m molecular_force_field.test.test_ase_calculator --n-atoms 50 --n-steps 100
```

- **优点**：与 ASE 直接集成，速度与 ASE 测试一致（~17 ms/步）。
- **缺点**：无 LAMMPS 的 fix、ensemble、输出格式等。

---

### 方案 2：pair_style mliap unified（需改模型）

**适用**：必须用 LAMMPS，且可接受模型改动。

详见 `LAMMPS_MLIAP_README.md`。要点：

1. **模型限制**：仅 `e3nn_layers`、`e3nn_layers_channelwise`、`cue_layers_channelwise`、`pure_cartesian_ictd_layers`、`pure_cartesian_ictd_layers_full` 支持 ML-IAP（因其支持 `precomputed_edge_vec` / edge forces）。其中 `cue_layers_channelwise`（spherical-save-cue）使用 cuEquivariance 实现 GPU 加速。
2. **LAMMPS 编译**：`PKG_ML-IAP=ON`、`MLIAP_ENABLE_PYTHON=ON`、`PKG_PYTHON=ON`。
3. **预期**：ML-IAP 接口为 ML 势设计，数据传递更高效，可显著减少 Python 回调开销。

---

### 方案 3：LibTorch C++ pair style（最快，开发量大）

**适用**：对性能要求极高，可投入 C++ 开发。

1. 用 **TorchScript** 导出模型为 `.pt`。
2. 编写 LAMMPS **自定义 pair style**（C++），链接 libtorch。
3. 在 C++ 中直接调用模型，**完全绕过 Python**。

- **优点**：无 Python 开销，可接近原生 C++ 势的速度。
- **缺点**：需维护 C++ 代码，模型更新需同步导出。

---

### 方案 4：fix external 微优化（已做）

当前已做优化：

- 使用 `lmp.numpy.extract_atom` 减少数据复制（若可用）。
- 使用 `np.fromiter` 替代双重 Python 循环。
- 力批量写入（若 `f` 为 numpy 数组）。

这些只能小幅改善，无法消除 C→Python 回调的主要开销。

---

## 建议

| 场景 | 建议 |
|------|------|
| 日常 MD、结构优化 | 用 **ASE** |
| 必须用 LAMMPS（NPT、特殊 fix 等） | 优先实现 **ML-IAP unified** |
| 超大规模、生产环境 | 考虑 **LibTorch C++ pair style** |

---

## 参考

- `molecular_force_field/docs/LAMMPS_MLIAP_README.md`：ML-IAP 接入说明
- `molecular_force_field/interfaces/test_ase_calculator.py`：ASE 测试脚本
- `molecular_force_field/interfaces/test_lammps_integration.py`：LAMMPS 联机测试
