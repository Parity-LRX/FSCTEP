# `long_range` 与 `feature-spectral` 两条实现路径的严格数学描述

本文档只描述**当前代码实现**对应的数学对象与算子，不描述“理想化版本”或“未来计划中的版本”。

对应源码：

- `molecular_force_field/models/long_range.py`
- 外层调用：`molecular_force_field/models/pure_cartesian_ictd_layers_full.py`
- 外层调用：`molecular_force_field/models/cue_layers_channelwise.py`

本文主要覆盖两条路径：

1. `long_range_mode=reciprocal-spectral-v1`
2. `feature_spectral_mode=fft`

并附带说明 `reciprocal-spectral-v1` 的两个后端：

- `direct_kspace`
- `mesh_fft`

本文**不**把 `latent-coulomb` 当作主路径，只在必要时顺带提及。

---

## 0. 渲染兼容约定（重要）

本文件用于 Markdown 数学公式渲染（KaTeX/MathJax 风格）。为避免渲染器对少数命令支持不一致，本文件遵循以下约定：

- **避免使用** `\operatorname{...}` 自定义算子名（某些渲染器对其行为不一致）。
- 需要“函数名”的地方统一用 `\mathrm{...}`（例如 `\mathrm{clip}`、`\mathrm{mod}`、`\mathrm{softplus}`）。
- 避免在 display 公式里写“更新/赋值箭头”；能写成定义（`:={}`）就写成定义。

为保证严格性，下面把本文使用到的几个非标准函数明确写成数学定义（后文均按此定义引用）：

- **softplus**：
  \[
  \mathrm{softplus}(x) := \log(1+e^x).
  \]
- **clip**（截断到区间）：
  \[
  \mathrm{clip}(x;\,a,b) := \min\{\max\{x,a\},\,b\}.
  \]
- **mod**（对整数取余；本文仅在整数索引上使用）：
  \[
  \mathrm{mod}(n,S) := n - S\Bigl\lfloor \frac{n}{S}\Bigr\rfloor,\qquad n\in\mathbb Z,\ S\in\mathbb N.
  \]
- **实部**：
  \[
  \mathrm{Re}(z) := \frac{z+\overline z}{2}.
  \]

## 1. 统一记号与实现约定

### 1.1 图、原子、批次

设一个 batch 中共有 \(B\) 个图，第 \(g\) 个图的原子集合记为
\[
\mathcal I_g \subset \{1,\dots,N\}, \qquad n_g := |\mathcal I_g|.
\]

代码中的 `batch[i]=g` 表示原子 \(i\) 属于图 \(g\)。

### 1.2 晶胞矩阵的行向量约定

对每个图 \(g\)，记其晶胞矩阵为
\[
A_g \in \mathbb R^{3\times 3}.
\]

在当前实现中，位置向量采用**行向量**约定，分数坐标定义为
\[
f_i = r_i A_{g(i)}^{-1},
\]
即源码中的 `frac = pos @ inv(cell)`。

因此反过来有
\[
r_i = f_i A_{g(i)}.
\]

同理，倒空间波矢也按行向量写为
\[
k = 2\pi\, m A_g^{-1},
\]
其中 \(m\in\mathbb Z^3\) 或离散 FFT 整数频率格点。

### 1.3 通道记号

- 对 `long_range` 路径，latent reciprocal source 的通道数记为
  \[
  C_s.
  \]
- 对 `feature-spectral` 路径，瓶颈通道数记为
  \[
  C_{\mathrm{lr}}.
  \]

若无歧义，向量内积一律写为
\[
\langle u,v\rangle = \sum_c u_c v_c.
\]

---

## 2. 边界条件与有效晶胞

### 2.1 `periodic`

对 `periodic`，实现中使用
\[
f_i = r_i A_g^{-1}, \qquad
\tilde f_i = f_i - \lfloor f_i \rfloor,
\]
即三个方向都做模 \(1\) 包裹。

有效晶胞为
\[
A_g^{\mathrm{eff}} = A_g.
\]

### 2.2 `slab`

对 `slab`，设 `slab_padding_factor = p \in \mathbb N, p\ge 1`。

实现中先计算未包裹分数坐标
\[
f_i = r_i A_g^{-1}.
\]

然后只对前两维做周期包裹：
\[
\tilde f_{i,x} = f_{i,x} - \lfloor f_{i,x} \rfloor, \qquad
\tilde f_{i,y} = f_{i,y} - \lfloor f_{i,y} \rfloor.
\]

第三维按如下规则映射到 padding 后的有效盒子内部：
\[
\tilde f_{i,z} = \frac{f_{i,z}}{p} + \frac{p-1}{2p}.
\]

实现中的有效晶胞为
\[
A_g^{\mathrm{eff}} =
\begin{bmatrix}
a_{g,1}\\
a_{g,2}\\
p\, a_{g,3}
\end{bmatrix},
\]
也即仅把第三个晶格向量乘以 \(p\)。

### 2.3 说明

因此，`slab` 在当前实现中的严格含义是：

- \(x,y\) 方向做周期边界；
- \(z\) 方向通过“有效晶胞放大 + 坐标居中映射”处理成带真空 padding 的 FFT 盒子；
- 在 mesh 索引级别，\(z\) 方向**不做模 \(S\)**，而是后续采用截断（clamp）边界，见第 3.3 节。

---

## 3. 共享的 CIC 网格投影与回插算子

以下算子被两条路径共同使用：

- `long_range(mesh_fft)`
- `feature-spectral(fft)`

设均匀网格尺寸为
\[
S := \texttt{mesh\_size}.
\]

### 3.1 八角点偏移与局部坐标

对任一原子 \(i\)，定义
\[
u_i = S\, \tilde f_i \in \mathbb R^3,
\qquad
b_i = \lfloor u_i \rfloor \in \mathbb Z^3,
\qquad
\alpha_i = u_i - b_i \in [0,1)^3.
\]

定义 8 个角点偏移
\[
\sigma \in \{0,1\}^3.
\]

### 3.2 CIC 权重

对 \(\sigma=(\sigma_x,\sigma_y,\sigma_z)\in\{0,1\}^3\)，定义 trilinear / CIC 权重
\[
w_{i,\sigma}
=
\prod_{d\in\{x,y,z\}}
(1-\alpha_{i,d})^{1-\sigma_d}\,
\alpha_{i,d}^{\sigma_d}.
\]

（当 \(\sigma_d=0\) 取 \(1-\alpha_{i,d}\)，当 \(\sigma_d=1\) 取 \(\alpha_{i,d}\)。）这正对应源码 `_corner_weights_from_frac()` 生成的 8 个权重。

### 3.3 mesh 边界映射

定义索引边界映射 \(\mathcal B\)（对应源码 `_apply_mesh_boundary`）。

对 `periodic`：
\[
\mathcal B_{\mathrm{per}}(n_x,n_y,n_z)
=
\left(
\mathrm{mod}(n_x,S),\;
\mathrm{mod}(n_y,S),\;
\mathrm{mod}(n_z,S)
\right).
\]

对 `slab`：
\[
\mathcal B_{\mathrm{slab}}(n_x,n_y,n_z)
=
\left(
\mathrm{mod}(n_x,S),\;
\mathrm{mod}(n_y,S),\;
\mathrm{clip}(n_z;\,0,S-1)
\right).
\]

也即：

- `periodic`：三维都做模 \(S\)（`torch.remainder`）；
- `slab`：前两维做模 \(S\)，第三维做区间 \([0,S-1]\) 截断（`clamp`）。

### 3.4 source 上网格（spread）

若原子 \(i\) 携带通道向量 \(x_i\in\mathbb R^C\)，则其 spread 到 3D 网格
\[
M \in \mathbb R^{S\times S\times S\times C}
\]
的规则定义为
\[
M(q)
:=
\sum_{i=1}^N
\sum_{\sigma\in\{0,1\}^3:\; q=\mathcal B(b_i+\sigma)}
w_{i,\sigma}\, x_i.
\]

### 3.5 mesh 回插（gather）

给定网格场
\[
\Phi \in \mathbb R^{S\times S\times S\times C},
\]
其回插到原子 \(i\) 的值定义为
\[
(\mathcal G\Phi)_i
=
\sum_{\sigma\in\{0,1\}^3}
w_{i,\sigma}\,
\Phi\bigl(\mathcal B(b_i+\sigma)\bigr).
\]

实现中 gather 与 spread 使用**同一套** CIC 权重与同一套边界映射。

---

## 4. 共享的离散频率与 FFT 约定

### 4.1 离散整数频率

当前实现通过
`torch.fft.fftfreq(S, d=1/S)`
生成一维频率，因此每个方向的频率集合是长度为 \(S\) 的整数型离散集合
\[
\Omega_S
=
\left\{
0,1,\dots,\left\lfloor \frac{S-1}{2}\right\rfloor,
-\left\lfloor \frac{S}{2}\right\rfloor,\dots,-1
\right\}.
\]

于是 3D 频率格点为
\[
m=(m_x,m_y,m_z)\in \Omega_S^3.
\]

### 4.2 波矢与模长

对任一图 \(g\)，定义
\[
k_g(m) = 2\pi\, m (A_g^{\mathrm{eff}})^{-1},
\qquad
\kappa_g(m) = \|k_g(m)\|_2.
\]

有效体积定义为
\[
V_g^{\mathrm{eff}} = |\det(A_g^{\mathrm{eff}})|.
\]

### 4.3 FFT 归一化

源码使用 `torch.fft.fftn` 与 `torch.fft.ifftn` 的默认规范：

- 正向 FFT **不归一化**；
- 逆向 FFT 含总格点数 \(S^3\) 的归一化。

因此本文所有“FFT/逆 FFT”都应按 PyTorch 默认离散约定理解，而不是按其它软件包的归一化约定理解。

---

## 5. `long_range_mode=reciprocal-spectral-v1` 的外层 latent source 定义

### 5.1 latent source head

给定每个原子的 invariant feature
\[
h_i \in \mathbb R^F,
\]
定义 latent source head
\[
s_i = W_2\, \mathrm{SiLU}(W_1 h_i + b_1) + b_2
\in \mathbb R^{C_s}.
\]

这对应 `LatentSourceHead`：

- 第一层线性：`Linear(F, hidden_dim)`
- 激活：`SiLU`
- 第二层线性：`Linear(hidden_dim, C_s)`

### 5.2 graph 内中和（neutralization）

若 `neutralize=True`，则对每个图 \(g\) 计算
\[
\bar s_g = \frac{1}{n_g}\sum_{i\in\mathcal I_g} s_i,
\]
并用
\[
\tilde s_i = s_i - \bar s_{g(i)}
\]
替代 \(s_i\) 进入后续 reciprocal kernel。

若 `neutralize=False`，则 \(\tilde s_i=s_i\)。

后文统一把进入 reciprocal kernel 的 source 记为 \(\tilde s_i\)。

---

## 6. `reciprocal-spectral-v1` 的 `direct_kspace` 后端

这一后端对应类 `ReciprocalSpectralKernel3D`，且**只支持** `boundary="periodic"`。

### 6.1 截断整数 \(k\)-格点

设 `kmax = K`。实现使用整数格点
\[
\mathcal M_K
=
\{-K,-K+1,\dots,K\}^3.
\]

若 `include_k0=False`，则去掉
\[
m=(0,0,0).
\]

### 6.2 原子相位

对图 \(g\) 中原子 \(i\)，分数坐标为
\[
\tilde f_i = r_i A_g^{-1} - \lfloor r_i A_g^{-1}\rfloor.
\]

对任意 \(m\in\mathcal M_K\)，相位定义为
\[
\theta_i(m) = 2\pi\, \tilde f_i \cdot m.
\]

### 6.3 按通道的 cosine/sine 结构因子

对每个图 \(g\)、格点 \(m\)、通道 \(c\)，定义
\[
C_g(m,c) = \sum_{i\in\mathcal I_g} \tilde s_{i,c}\cos\theta_i(m),
\]
\[
S_g(m,c) = \sum_{i\in\mathcal I_g} \tilde s_{i,c}\sin\theta_i(m).
\]

### 6.4 频谱权重

此后端不使用 `green_mode`；它始终使用 `RadialSpectralFilter`。

设
\[
\varepsilon = \texttt{k\_norm\_floor} > 0,
\qquad
\kappa = \kappa_g(m).
\]

先定义
\[
\kappa_\varepsilon = \max\{\kappa,\varepsilon\}.
\]

然后构造三维输入特征
\[
\xi(\kappa)
=
\left(
\log(1+\kappa_\varepsilon),\;
\bigl[\log(1+\kappa_\varepsilon)\bigr]^2,\;
\kappa_\varepsilon^{-1}
\right)
\in \mathbb R^3.
\]

令 \(f_{\mathrm{rad}}\) 为一个 3 层 MLP：
\[
\mathbb R^3 \to \mathbb R^{H} \to \mathbb R^{H} \to \mathbb R,
\]
中间激活均为 `SiLU`。则 learned scale 为
\[
\lambda(\kappa) = \mathrm{softplus}\!\bigl(f_{\mathrm{rad}}(\xi(\kappa))\bigr) > 0.
\]

基核为
\[
g_{\mathrm{C}}(\kappa) = \frac{4\pi}{\kappa_\varepsilon^2}.
\]

最终频谱权重为
\[
W_g(m) = g_{\mathrm{C}}(\kappa_g(m))\, \lambda(\kappa_g(m)).
\]

若 `include_k0=False`，实现中还会执行
\[
W_g(m)=0 \quad \text{当 } \kappa_g(m)\le \varepsilon.
\]

### 6.5 每原子 potential

对图 \(g\) 中原子 \(i\)，定义
\[
\phi_i
=
\frac{1}{V_g}
\sum_{m\in\mathcal M_K}
W_g(m)
\Bigl(
C_g(m)\cos\theta_i(m) + S_g(m)\sin\theta_i(m)
\Bigr)
\in \mathbb R^{C_s},
\]
其中
\[
V_g = |\det(A_g)|.
\]

这里 \(C_g(m)\)、\(S_g(m)\)、\(\phi_i\) 都是 \(C_s\)-维向量，上式按通道逐分量成立。

### 6.6 `potential` 能量分配

若 `energy_partition="potential"`，则每原子能量为
\[
e_i^{\mathrm{dir}}
=
\frac12 \langle \tilde s_i, \phi_i\rangle.
\]

### 6.7 `uniform` 能量分配

若 `energy_partition="uniform"`，则实现先计算图总能量
\[
E_g^{\mathrm{dir}}
=
\frac{1}{2V_g}
\sum_{m\in\mathcal M_K}
W_g(m)
\sum_{c=1}^{C_s}
\Bigl(
C_g(m,c)^2 + S_g(m,c)^2
\Bigr),
\]
然后均匀分给图中每个原子：
\[
e_i^{\mathrm{dir}} = \frac{E_g^{\mathrm{dir}}}{n_g}.
\]

### 6.8 外层额外缩放

当 `reciprocal_backend="direct_kspace"` 时，`LatentReciprocalLongRange` 还会乘一个可学习标量
\[
\alpha_{\mathrm{lr}} \in \mathbb R,
\]
故最终输出为
\[
e_i = \alpha_{\mathrm{lr}}\, e_i^{\mathrm{dir}}.
\]

源码中该参数初始化为 \(0\)。

---

## 7. `reciprocal-spectral-v1` 的 `mesh_fft` 后端

这一后端对应类 `MeshLongRangeKernel3D`，支持

- `boundary="periodic"`
- `boundary="slab"`

且是当前 long-range 主路线。

### 7.1 source 上网格

对图 \(g\)，将所有 \(\tilde s_i \in \mathbb R^{C_s}\) 经第 3 节的 `spread` 算子映射为
\[
M_g \in \mathbb R^{S\times S\times S\times C_s}.
\]

### 7.2 Green kernel

设 \(\kappa=\kappa_g(m)\)，\(\varepsilon=\texttt{k\_norm\_floor}\)，\(\kappa_\varepsilon=\max\{\kappa,\varepsilon\}\)。

#### 7.2.1 `green_mode="poisson"`

定义
\[
G_g(m) = \frac{4\pi}{\kappa_\varepsilon^2}.
\]

#### 7.2.2 `green_mode="learned_poisson"`

仍使用第 6.4 节中 `RadialSpectralFilter` 的定义：
\[
G_g(m) = \frac{4\pi}{\kappa_\varepsilon^2}\,\lambda(\kappa_g(m)),
\]
其中 \(\lambda(\kappa)>0\) 由 MLP + `softplus` 给出。

#### 7.2.3 \(k=0\) 处理

若 `include_k0=False`，则实现中强制
\[
G_g(m)=0 \quad \text{当 } \kappa_g(m)\le \varepsilon.
\]

### 7.3 频域求势

实现中使用的频域权重为
\[
\widetilde G_g(m) = \frac{G_g(m)}{V_g^{\mathrm{eff}}}.
\]

因此 mesh potential 由
\[
\widehat{\Phi}_g(m)
=
\widehat{M}_g(m)\, \widetilde G_g(m)
\]
给出，再取逆 FFT：
\[
\Phi_g = \mathrm{IFFT}\bigl(\widehat{\Phi}_g\bigr).
\]

由于输入 mesh 为实数，代码最后取
\[
\Phi_g \;:=\; \mathrm{Re}\!\bigl(\Phi_g\bigr).
\]

### 7.4 potential 回插

将 \(\Phi_g\) 经第 3 节的 gather 算子回插到原子，得
\[
\phi_i = (\mathcal G \Phi_g)_i \in \mathbb R^{C_s}.
\]

### 7.5 `potential` 能量分配

若 `energy_partition="potential"`，则
\[
e_i^{\mathrm{mesh}}
=
\frac12 \langle \tilde s_i, \phi_i\rangle.
\]

### 7.6 `uniform` 能量分配

若 `energy_partition="uniform"`，则实现先求
\[
E_g^{\mathrm{mesh}}
=
\sum_{i\in\mathcal I_g}
\frac12 \langle \tilde s_i, \phi_i\rangle,
\]
再平均分给图中原子：
\[
e_i^{\mathrm{mesh}} = \frac{E_g^{\mathrm{mesh}}}{n_g}.
\]

### 7.7 与 `direct_kspace` 的严格差异

当前实现中，`mesh_fft` 与 `direct_kspace` 的严格差异包括：

1. `mesh_fft` 通过 `green_mode` 选择 `poisson` 或 `learned_poisson`；
2. `direct_kspace` 不读取 `green_mode`，而始终使用 `RadialSpectralFilter`；
3. `mesh_fft` 使用真实 FFT 网格与 CIC 赋值；
4. `direct_kspace` 直接在截断整数 \(k\)-格点上显式求和；
5. `mesh_fft` 输出**没有**额外的全局可学习能量缩放；
6. `direct_kspace` 输出会乘一个额外的可学习标量 \(\alpha_{\mathrm{lr}}\)。

### 7.8 source 导出与初始化说明

当 `reciprocal_backend="mesh_fft"` 时，外层模块 `LatentReciprocalLongRange`：

- 会设置 `exports_reciprocal_source=True`
- 且把 source head 最后一层线性参数初始化为零

即初始化时
\[
W_2 = 0,\qquad b_2 = 0.
\]

因此初始时
\[
s_i = 0,\qquad \tilde s_i = 0,\qquad e_i^{\mathrm{mesh}} = 0.
\]

若调用 `forward(..., return_source=True)`，返回的 source 是
\[
s_i,
\]
即**中和前**的 source，而不是 \(\tilde s_i\)。

---

## 8. `feature_spectral_mode=fft` 的严格定义

对应类：

- `FeatureSpectralResidualBlock`
- `FeatureSpectralFilterGrid`

这是一个**特征空间**的低秩 FFT 残差块，而不是能量意义下的 Poisson / Ewald 求解器。

### 8.1 输入特征

设每个原子的 invariant feature 为
\[
h_i \in \mathbb R^F.
\]

先做层归一化：
\[
\bar h_i = \mathrm{LayerNorm}(h_i).
\]

再做线性降维：
\[
z_i = W_{\mathrm{in}} \bar h_i + b_{\mathrm{in}}
\in \mathbb R^{C_{\mathrm{lr}}}.
\]

这里
\[
C_{\mathrm{lr}} = \texttt{feature\_spectral\_bottleneck\_dim}.
\]

### 8.2 graph 内零均值

若 `feature_spectral_neutralize=True`，则对每个图 \(g\) 定义
\[
\bar z_g = \frac{1}{n_g}\sum_{i\in\mathcal I_g} z_i,
\qquad
\tilde z_i = z_i - \bar z_{g(i)}.
\]

否则 \(\tilde z_i=z_i\)。

### 8.3 bottleneck source 上网格

对图 \(g\)，使用第 3 节完全相同的 CIC spread，把 \(\tilde z_i\) 映射成
\[
M_g \in \mathbb R^{S\times S\times S\times C_{\mathrm{lr}}}.
\]

### 8.4 频域滤波器

对每个频率格点 \(m\in\Omega_S^3\)，先构造
\[
\kappa = \kappa_g(m), \qquad
\kappa_\varepsilon = \max\{\kappa,\varepsilon\}.
\]

和 long-range 的 `RadialSpectralFilter` 一样，定义
\[
\xi(\kappa)
=
\left(
\log(1+\kappa_\varepsilon),\;
\bigl[\log(1+\kappa_\varepsilon)\bigr]^2,\;
\kappa_\varepsilon^{-1}
\right)
\in \mathbb R^3
\]
\[
\lambda(\kappa) = \mathrm{softplus}\!\bigl(f_{\mathrm{rad}}(\xi(\kappa))\bigr) > 0
\]
\[
\rho_g(m)
=
\frac{4\pi}{\kappa_\varepsilon^2}\,\lambda(\kappa_g(m)).
\]

若 `feature_spectral_include_k0=False`，则
\[
\rho_g(m)=0 \quad \text{当 } \kappa_g(m)\le \varepsilon.
\]

### 8.5 通道缩放

此外，每个 bottleneck 通道 \(c\) 还有一个独立正缩放：
\[
\eta_c = \mathrm{softplus}(\beta_c) > 0.
\]

因此频域中对每个通道的总滤波因子为
\[
\rho_g(m)\,\eta_c.
\]

### 8.6 频域滤波

对网格场的 FFT 记为
\[
\widehat{M}_g(m,c).
\]

则过滤后的频域张量定义为
\[
\widehat{M}^{\,\prime}_g(m,c)
=
\widehat{M}_g(m,c)\,\rho_g(m)\,\eta_c.
\]

然后取逆 FFT：
\[
M^{\,\prime}_g = \mathrm{IFFT}\bigl(\widehat{M}^{\,\prime}_g\bigr),
\]
并取实部：
\[
M^{\,\prime}_g \;:=\; \mathrm{Re}\!\bigl(M^{\,\prime}_g\bigr).
\]

### 8.7 回插与残差

将 \(M^{\,\prime}_g\) 用第 3 节的 gather 算子回插，得到
\[
z^{\,\prime}_i = (\mathcal G M^{\,\prime}_g)_i \in \mathbb R^{C_{\mathrm{lr}}}.
\]

然后上投影回原特征维度：
\[
r_i = W_{\mathrm{out}} z^{\,\prime}_i \in \mathbb R^F.
\]

注意 `out_proj` 无 bias，因此这里没有额外偏置项。

### 8.8 残差门控

设可学习标量门控参数为 \(g\in\mathbb R\)，则实现中的有效门控是
\[
\gamma = \tanh(g)\in(-1,1).
\]

最终输出特征为
\[
h_i^{\mathrm{out}} = h_i + \gamma\, r_i.
\]

### 8.9 返回的 auxiliary source

`FeatureSpectralResidualBlock.forward()` 返回一个二元组：

\[
\bigl(h_i^{\mathrm{out}},\; z_i\bigr).
\]

注意第二项是
\[
z_i = W_{\mathrm{in}}\mathrm{LayerNorm}(h_i)+b_{\mathrm{in}},
\]
即**中和前**、**滤波前**的 bottleneck source，而不是 \(\tilde z_i\)、也不是 \(z^{\,\prime}_i\)。

---

## 9. 两条路径的本质差异

### 9.1 `long_range(mesh_fft)` 是“source-to-energy”路径

其严格输入输出关系是：

1. 从 invariant feature 生成 latent reciprocal source \(\tilde s_i\)；
2. 通过网格、Green kernel、频域卷积得到 potential \(\phi_i\)；
3. 再由
   \[
   e_i = \frac12 \langle \tilde s_i,\phi_i\rangle
   \]
   构造每原子 long-range 能量。

因此它是**能量路径**。

### 9.2 `feature-spectral(fft)` 是“feature-to-feature”路径

其严格输入输出关系是：

1. 从 feature \(h_i\) 压缩得到 bottleneck feature \(z_i\)；
2. 对 \(z_i\) 做网格-FFT-回插；
3. 得到残差 \(r_i\)；
4. 用
   \[
   h_i^{\mathrm{out}} = h_i + \gamma r_i
   \]
   修正原特征。

因此它是**特征残差路径**，而不是显式能量核路径。

---

## 10. 与导出 / runtime 的严格对应

### 10.1 `long_range(mesh_fft)`

若 `LatentReciprocalLongRange` 使用 `mesh_fft`，则：

- `exports_reciprocal_source=True`
- 外层模型会把 returned source 当作 `reciprocal_source` 暴露给导出路径
- 导出元数据会记录：
  - `reciprocal_source_channels`
  - `reciprocal_source_boundary`
  - `reciprocal_source_slab_padding_factor`
  - `long_range_boundary`
  - `long_range_mesh_size`
  - `long_range_slab_padding_factor`
  - `long_range_reciprocal_backend`
  - `long_range_green_mode`

### 10.2 `feature-spectral(fft)`

若没有 `long_range(mesh_fft)`，但启用了 `feature-spectral(fft)`，则外层模型会把
\[
z_i
\]
作为 `reciprocal_source` 暴露给导出路径，并记录：

- `reciprocal_source_channels = C_{\mathrm{lr}}`
- `reciprocal_source_boundary = feature_spectral_boundary`
- `reciprocal_source_slab_padding_factor = feature_spectral_slab_padding_factor`

因此，从导出 / runtime 角度看，二者都可以向外提供一个“每原子 reciprocal source”，但其**数学意义不同**：

- `long_range(mesh_fft)` 导出的 source 是用于 long-range 能量核的 latent source；
- `feature-spectral(fft)` 导出的 source 是 feature bottleneck 的 pre-neutralization 变量。

---

## 11. 当前实现中必须明确说明的限制

### 11.1 `green_mode` 只严格作用于 `mesh_fft`

当前实现中：

- `green_mode` 只在 `MeshLongRangeKernel3D` 中生效；
- `direct_kspace` 后端不读取 `green_mode`，而始终使用 `RadialSpectralFilter`。

因此任何数学描述若把 `green_mode` 同时施加到 `direct_kspace`，都是**不严格**的。

### 11.2 `assignment` 当前只支持 `cic`

当前实现只允许
\[
\texttt{assignment}=\texttt{cic}.
\]

没有 TSC、PCS 或其它更高阶赋值方案。

### 11.3 `slab` 的 \(z\) 方向不是周期包裹

`slab` 下 mesh 索引的第三维是 clamp，而不是模 \(S\)。

因此数学上必须写成第 3.3 节的
\[
\mathcal B_{\mathrm{slab}}
\]
而不能把它误写成 3D 周期 FFT 盒子。

### 11.4 `feature-spectral` 不是 Poisson 求能

虽然 `feature-spectral` 也使用
\[
\frac{4\pi}{\kappa^2}
\]
型基核乘 learned radial filter，但它的输出是**特征残差**而不是
\[
\frac12 \langle \text{source}, \text{potential} \rangle
\]
型能量。

因此不能把 `feature-spectral` 写成“LES 能量核”的特例；两者在当前代码中是不同算子。

### 11.5 `latent-coulomb + tree_fmm` 是开放边界实空间近似

新增的 `long_range_mode=latent-coulomb` + `long_range_backend=tree_fmm` 不再经过第 3–7 节的 FFT / reciprocal 管线。

它的当前严格语义是：

- `boundary="nonperiodic"`；
- source 为每原子标量 latent charge；
- 势函数仍取 screened Coulomb / Yukawa 型
  \[
  G(r)=\frac{e^{-\kappa r}}{r};
  \]
- 对足够远的 cluster，用 Barnes-Hut 风格的 monopole 近似替代逐对求和；
- `multipole_order=0`，因此当前还不是完整高阶 FMM，而是 correctness-first 的 treecode / FMM 首版；
- 在 USER-MFFTORCH runtime 中，单 rank 路径仍是本地树近似；多 rank 路径会对远处 rank 交换 coarse multipole / cluster 摘要、对近邻 rank 交换精确原子 source，因此语义上是开放边界 distributed tree/FMM 首版，而不是 allgather 全原子的 replicated solver。

若导出到 runtime，则 `core.pt` 只导出 source，不在 TorchScript 图中显式执行树求和；真正的 open-boundary tree 求势由 runtime 侧继续完成。

---

## 12. 实现到数学的最小一一对应

### 12.1 共享基础函数

- `_effective_cell_for_boundary`：第 2 节的 \(A_g^{\mathrm{eff}}\)
- `_prepare_frac_for_boundary`：第 2 节的 \(\tilde f_i\)
- `_corner_weights_from_frac`：第 3.2 节的 \(w_{i,\sigma}\)
- `_apply_mesh_boundary`：第 3.3 节的 \(\mathcal B\)
- `_spread_source_to_mesh`：第 3.4 节的 spread
- `_gather_source_from_mesh`：第 3.5 节的 gather

### 12.2 `long_range`

- `LatentSourceHead`：第 5.1 节
- `DenseRealSpaceLongRangeKernel`：`latent-coulomb` 的 dense pairwise 实空间核
- `OpenBoundaryTreeLongRangeKernel`：第 11.5 节的开放边界树近似
- `LatentCoulombLongRange`：实空间 latent-charge 外层封装（dense / tree 两个 backend）
- `ReciprocalSpectralKernel3D`：第 6 节
- `ReciprocalGreenKernel`：第 7.2 节
- `MeshLongRangeKernel3D`：第 7 节
- `LatentReciprocalLongRange`：第 5 节外层封装

### 12.3 `feature-spectral`

- `FeatureSpectralFilterGrid`：第 8.4–8.6 节
- `FeatureSpectralResidualBlock`：第 8 节整体

---

## 13. 一句话总结

严格按照当前实现：

- `long_range(reciprocal-spectral-v1, mesh_fft)` 是
  \[
  \text{feature} \to \text{latent source} \to \text{mesh reciprocal potential} \to \text{atom energy}
  \]
  的路径；
- `feature-spectral(fft)` 是
  \[
  \text{feature} \to \text{bottleneck source} \to \text{spectral mesh filter} \to \text{feature residual}
  \]
  的路径；
- 两者共享同一套 `periodic/slab + CIC + FFT` 网格机制，但**最终语义不同**：
  一个输出能量，一个输出特征残差。
