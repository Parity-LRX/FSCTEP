## Pure-Cartesian Tensor Product的数学描述

### 与主文档 `PAPER_CROSS_LAYER_FUSION_THEORY.md` 的记号对齐（重要）
本文档作为同一论文的“实现-数学附录”，其核心对象与主文档第 1 节 Definition 3–4 的等变表示/等变双线性算子一致。为避免两份文档之间的符号歧义，现给出**明确的符号对照表**：

- **群**：主文档在主体推导中多取 \(G=\mathrm{SO}(3)\)；本文档为处理反射/宇称并与实现一致，常使用 \(O(3)\)（见第 2 节），并在需要时指出限制到 \(SO(3)\) 的特例（即 \(\det R=+1\)）。
- **角动量/不可约阶**：主文档使用 \(l=0,\dots,l_{\max}\) 表示不可约表示的阶（维数 \(2l+1\)）。本文档中若出现 \(\ell\) 作为不可约阶，则应理解为主文档中的 \(l\)（两者一一对应）。
- **pure-Cartesian 的张量秩（rank）**：主文档为避免与网络深度 \(L\) 混淆，使用 \(\ell\) 表示 Cartesian 张量秩（rank）。本文档沿用记号 \(L\) 表示该秩（例如 \(\mathcal T_L=(\mathbb R^3)^{\otimes L}\)、维数 \(3^L\)）。二者对应关系如下：
  \[
  \text{主文档的 Cartesian rank } \ell\quad \longleftrightarrow\quad \text{本文档的 } L.
  \]
  本文在不修改既有推导结构的前提下保留 \(L\) 记号；如需与主文档完全一致，可据此做一一替换。

本文阐述在**全笛卡尔 \(3^L\)** 表示下，使用 **Kronecker δ** 与 **Levi-Civita ε** 构造**严格 O(3) 等变**张量积的方法，及其与 ICTD（Irreducible Cartesian Tensor Decomposition，不可约笛卡尔张量分解）[3]/trace-chain（\((2\ell+1)\) irreps）的关系，以及本实现提供的"无球谐 irreps 内部表示"版本。


- **第 1–7 节**：`pure-cartesian` / `pure-cartesian-sparse` 的严格 O(3) 版本，内部表示 \(\bigoplus_L (\mathbb R^3)^{\otimes L}\)（维度 \(3^L\)），显式 true/pseudo（\(\mathbb Z_2\) 分级）处理 \(\det(R)\)。
- **第 8 节**：总结（实现到数学的一一对应）
- **第 9 节**：irreps（如 `64x0e+64x1o+64x2e`）与 pure-cartesian（\(3^L\)）的对应关系。
- **第 10 节**：`ictd_fast.py` 的快速不可约投影（对称子空间的 trace-chain / STF），从对称笛卡尔张量提取 \((2\ell+1)\) 块坐标，属于 SO(3) 范畴。
- **第 11 节**：`ictd_irreps.py` / `pure_cartesian_ictd_layers.py` 的 ICTD-irreps 内部表示。其数学构造可表述为在调和多项式空间上的交织算子（见第 11.0–11.7 节；限制到 \(SO(3)\) 时与标准 irreps 观点一致）。具体实现为数值线性代数过程，故在实验中以“数值等变误差”（第 12.0 节）报告其偏差量级。

> 备注（Implementation mapping）：
> 本附录中的对象与工程实现的对应关系如下（用于复现实验与核对符号；不参与数学证明）：
> - `molecular_force_field/models/pure_cartesian.py`：
>   - `PureCartesianTensorProductO3`（第 5 节）
>   - `PureCartesianTensorProductO3Sparse`（第 6 节）
>   - `_enumerate_paths`, `_enumerate_paths_sparse`, `_einsum_for_path`（第 4 节的路径算子 \(\Gamma_{k,\epsilon}\) 的坐标实现）
>   - `split_by_rank_o3`, `merge_by_rank_o3`, `rotate_rank_tensor`, `edge_rank_powers`（第 2、7 节）
> - `molecular_force_field/models/pure_cartesian_layers.py` / `pure_cartesian_sparse_layers.py`：
>   - 以消息传递方式调用上述张量积（第 7 节的边几何张量 + 第 5/6 节的双线性耦合）

### 阅读导航

- **严格 O(3)（含宇称）纯笛卡尔张量积**：第 1–6 节（重点：第 5–6 节）
- **Sparse 模式的稀疏化策略及 `k_policy/allow_epsilon/assume_pseudo_zero` 的数学含义**：第 6 节
- **`64x0e+64x1o+64x2e` 与 \(3^L\) 的对应关系**：第 9 节
- **快速 ICTD 投影（从对称张量提取 irreps 块）**：第 10 节
- **无球谐 irreps 内部表示及通过多项式构造 CG 张量**：第 11 节

### 符号约定（贯穿全文）
- **irreps**：不可约表示（irreducible representation）的缩写，符号来自 e3nn 库 [1]。在本文语境下，`irreps` 指 \(O(3)\)（或其子群 \(SO(3)\)）的有限维实表示的直和分解。e3nn 的单个不可约块可记为 \((l,p)\)，其中
  - \(l\in\mathbb N\) 为不可约阶；限制到 \(SO(3)\) 时该块维数为 \(2l+1\)，表示矩阵记为 \(D^{(l)}(R)\)；
  - \(p\in\{+1,-1\}\) 为 **宇称标签**（parity label），定义为全反演 \(-I\in O(3)\) 在该块上的作用：\(D^{(l,p)}(-I)=p\,I_{2l+1}\)。
  
  在字符串记号中，`e` 对应 \(p=+1\)，`o` 对应 \(p=-1\)。例如 `64x1o` 表示 64 个 \((l=1,p=-1)\) 的块直和。
- **\(L\)**：pure-cartesian 的"秩/阶"（rank），对应全笛卡尔表示 \((\mathbb R^3)^{\otimes L}\)，维度 \(3^L\)。
- **\(\ell\)**：不可约表示/调和多项式的"阶"（irrep degree / harmonic degree），对应维度 \(2\ell+1\) 的 irreps 块。
- **\(s\in\{0,1\}\)**：\(\mathbb Z_2\) 分级（true/pseudo），实现严格 \(O(3)\)（含反射）等变性的标记机制（见第 2 节）。
- **\(R\in O(3)\)**：正交变换；\(\det(R)\in\{\pm1\}\)。\(R\in SO(3)\) 表示仅旋转（\(\det=+1\)）。

---

## 1. 记号与基本对象

### 1.1 O(3) 与 det 因子
令
\[
O(3)=\{R\in\mathbb R^{3\times 3}\mid R^\top R=I\},
\quad \det(R)\in\{+1,-1\}.
\]
\(\det(R)=-1\) 表示包含反射（宇称）变换。

### 1.2 全笛卡尔秩张量空间（\(3^L\) 分量）
对每个秩 \(L\in\mathbb N\)，定义
\[
\mathcal T_L := (\mathbb R^3)^{\otimes L},
\qquad \dim(\mathcal T_L)=3^L.
\]
当 \(L=0\) 时 \(\mathcal T_0=\mathbb R\)（标量）。

给定通道数 \(C\)，秩 \(L\) 的特征块取
\[
\mathbb R^C\otimes \mathcal T_L.
\]

---

## 2. O(3) 的自然作用与 Z2 分级（true / pseudo）

### 2.1 自然张量表示（对秩指标）
对 \(T\in\mathcal T_L\)（坐标表示为 \(T_{i_1\cdots i_L}\)），定义
\[
(R\cdot T)_{i_1\cdots i_L}
=\sum_{j_1,\ldots,j_L}
R_{i_1j_1}\cdots R_{i_Lj_L}\,T_{j_1\cdots j_L}.
\]
通道维 \(\mathbb R^C\) 视为标量拷贝，不随 \(R\) 变化。

### 2.2 O(3) 严格等变所需的 true/pseudo 分级
为在 \(O(3)\)（而非仅 \(SO(3)\)）下严格等变，引入 \(\mathbb Z_2\) 分级：
- \(s=0\)：true tensor（极张量 / polar tensor）
- \(s=1\)：pseudotensor（轴张量 / axial tensor）

定义对分级块 \(T^{(s)}\in\mathcal T_L\) 的群作用为
\[
(R\cdot T^{(s)}) := (\det R)^s\,(R\cdot T).
\]
> 备注（Implementation）：
> 该分级在实现中通过布尔标记（例如 `pseudo=s`）携带，并在旋转/反射作用时乘以 \((\det R)^s\)。

### 2.3 全特征空间（本层的输入/输出空间）
给定 \(C, L_{\max}\)，定义
\[
\mathcal V(C,L_{\max})
=\bigoplus_{s\in\{0,1\}}\ \bigoplus_{L=0}^{L_{\max}}\left(\mathbb R^C\otimes\mathcal T_L\right).
\]
> 备注（Implementation）：
> 工程实现一般将该直和空间“扁平化”为一个张量，并提供 `split_by_rank_o3/merge_by_rank_o3` 等打包与拆包算子以在各 rank/parity 块之间切换；该实现细节不影响本文的表示论结论。

---

## 3. δ 与 ε：唯一使用的几何不变张量

### 3.1 Kronecker δ
\[
\delta_{ij}=
\begin{cases}
1,& i=j\\
0,& i\ne j
\end{cases}
\]
满足 \(O(3)\) 不变性：
\[
\sum_{a,b} R_{ia}R_{jb}\delta_{ab}=\delta_{ij}.
\]

### 3.2 Levi-Civita ε
\(\varepsilon_{ijk}\) 为三维完全反对称张量，满足
\[
\sum_{a,b,c}R_{ia}R_{jb}R_{kc}\varepsilon_{abc}=(\det R)\,\varepsilon_{ijk}.
\]
因此一次 ε 收缩会引入额外的 \(\det R\) 因子（pseudo 的来源，见第 2.2 节）。

**等价定义：**
\[
\varepsilon_{ijk}=
\begin{cases}
 +1,& (i,j,k)\ \text{为}\ (1,2,3)\ \text{的偶置换}\\
 -1,& (i,j,k)\ \text{为}\ (1,2,3)\ \text{的奇置换}\\
 0,& \text{否则}.
\end{cases}
\]

> 备注（Implementation）：
> 实现中可通过显式常数张量返回 \(\varepsilon_{ijk}\)，其坐标满足式 (3.2) 的变换规律。

---

## 4. “路径”（path）与规范化指标收缩算子 Γ

令 \(V:=\mathbb R^3\) 带标准内积 \(\langle u,v\rangle=u^\top v\)，并记 \(\mathcal T_L:=V^{\otimes L}\)（第 1.2 节）。本节定义一族由 \(\delta\) 与 \(\varepsilon\) 生成的双线性收缩算子 \(\Gamma_p\)，它们是“pure-cartesian 路径枚举”的数学抽象，同时也是 \(O(3)\) 的（伪）交织算子（intertwiner）。

### 4.1 路径参数与输出秩
固定 \(L_1,L_2\in\{0,\dots,L_{\max}\}\)。一条路径 \(p\) 由
\[
p=(k,\epsilon),\qquad k\in\{0,\dots,\min(L_1,L_2)\},\ \epsilon\in\{0,1\},
\]
给出，其中 \(k\) 表示 \(\delta\) 成对收缩次数，\(\epsilon=1\) 表示额外使用一次 \(\varepsilon\) 收缩（\(\epsilon=0\) 表示不使用）。

定义输出秩
\[
L_{\mathrm{out}}:=L_1+L_2-2k-\epsilon,
\]
并要求 \(0\le L_{\mathrm{out}}\le L_{\max}\)。

> 备注（关于“规范化顺序”的数学意义）：
> 本文固定一种明确的指标约定（见 4.2 节）以给出确定的坐标公式；任何其它一致的指标选择/输出排列方案，都只会与本文定义相差一个输入/输出置换算子。置换算子与 \(O(3)\) 的自然张量作用可交换，因此不影响下文的等变性定理。

### 4.2 \(\Gamma_{k,\epsilon}\) 的严格定义（坐标形式）
设 \(A\in\mathcal T_{L_1}\), \(B\in\mathcal T_{L_2}\)。采用如下规范约定：
- 将 \(A\) 的最后 \(k\) 个指标与 \(B\) 的最后 \(k\) 个指标按相同求和指标成对收缩（\(\delta\) 收缩）。
- 若 \(\epsilon=1\)，则再取 \(A\) 剩余自由指标中的最后一个与 \(B\) 剩余自由指标中的最后一个，通过 \(\varepsilon\) 生成一个新的输出指标，并把该新指标放在输出指标串末尾。

分别给出两种情形的坐标公式。

**(i) 仅 \(\delta\) 收缩（\(\epsilon=0\)）**。当 \(\epsilon=0\) 时 \(L_{\mathrm{out}}=L_1+L_2-2k\)，定义
\[
\big(\Gamma_{k,0}(A,B)\big)_{i_1\ldots i_{L_1-k}\,j_1\ldots j_{L_2-k}}
:=\sum_{a_1,\ldots,a_k=1}^3
A_{i_1\ldots i_{L_1-k}\,a_1\ldots a_k}\;
B_{j_1\ldots j_{L_2-k}\,a_1\ldots a_k}.
\tag{4.1}
\]

**(ii) \(\delta\) 收缩后再用一次 \(\varepsilon\)（\(\epsilon=1\)）**。当 \(\epsilon=1\) 时 \(L_{\mathrm{out}}=L_1+L_2-2k-1\)，并要求 \(L_1-k\ge 1\) 且 \(L_2-k\ge 1\)。定义
\[
\begin{aligned}
&\big(\Gamma_{k,1}(A,B)\big)_{i_1\ldots i_{L_1-k-1}\,j_1\ldots j_{L_2-k-1}\,c}\\
&\qquad:=\sum_{a_1,\ldots,a_k=1}^3\ \sum_{u,v=1}^3
\varepsilon_{cuv}\;
A_{i_1\ldots i_{L_1-k-1}\,u\,a_1\ldots a_k}\;
B_{j_1\ldots j_{L_2-k-1}\,v\,a_1\ldots a_k}.
\end{aligned}
\tag{4.2}
\]

由此得到双线性算子
\[
\Gamma_{k,\epsilon}:\mathcal T_{L_1}\times \mathcal T_{L_2}\to \mathcal T_{L_{\mathrm{out}}}.
\]
下文亦写 \(\Gamma_p:=\Gamma_{k,\epsilon}\)。

### 4.3 \(\Gamma_{k,\epsilon}\) 的 \(O(3)\)（伪）等变性
### Lemma 1（Kronecker \(\delta\) 与 Levi-Civita \(\varepsilon\) 的变换律）
对任意 \(R\in O(3)\)，有
\[
\sum_{a,b}R_{ia}R_{jb}\delta_{ab}=\delta_{ij},\qquad
\sum_{a,b,c}R_{ia}R_{jb}R_{kc}\varepsilon_{abc}=(\det R)\varepsilon_{ijk}.
\tag{4.3}
\]

### Proposition 1（收缩算子 \(\Gamma_{k,\epsilon}\) 的（伪）等变性）
对任意 \(R\in O(3)\) 与任意 \(A\in\mathcal T_{L_1},B\in\mathcal T_{L_2}\)，成立
\[
\Gamma_{k,0}(R\cdot A,\ R\cdot B)=R\cdot\Gamma_{k,0}(A,B),
\tag{4.4}
\]
以及
\[
\Gamma_{k,1}(R\cdot A,\ R\cdot B)=(\det R)\, \big(R\cdot\Gamma_{k,1}(A,B)\big).
\tag{4.5}
\]

**Proof.** 以坐标表示直接验证。将第 2.1 节的自然张量作用展开代入式 (4.1) 与 (4.2)，并把求和中出现的 \(R\) 因子按收缩指标聚合：\(\delta\) 收缩处出现 \(\sum_{a}R_{\alpha a}R_{\beta a}\)，由式 (4.3) 第一式化为 \(\delta_{\alpha\beta}\)；\(\varepsilon\) 收缩处出现 \(\sum_{u,v,c}R_{\gamma c}R_{\alpha u}R_{\beta v}\varepsilon_{cuv}\)，由式 (4.3) 第二式化为 \((\det R)\varepsilon_{\gamma\alpha\beta}\)。整理后所有 \(R\) 仅作用于输出自由指标，得到式 (4.4) 与 (4.5)。□

### 4.4 示例（\(\delta\) 收缩）
取 \(L_1=2\)（矩阵）、\(L_2=1\)（向量）、\(k=1\)、\(\epsilon=0\)，则 \(L_{\mathrm{out}}=1\)，且
\[
\big(\Gamma_{1,0}(A,B)\big)_i=\sum_{m=1}^3 A_{im}B_m,
\]
即矩阵-向量乘法（把 \(A\) 视为二阶张量）。

---

## 5. Pure-Cartesian（严格 O(3)）张量积：`PureCartesianTensorProductO3`

### 5.1 输入输出与参数
设输入通道 \(C_1,C_2\)，输出通道 \(C_{\text{out}}\)。本层实现一个映射：
\[
\mathrm{TP}:\mathcal V(C_1,L_{\max})\times \mathcal V(C_2,L_{\max})\to \mathcal V(C_{\text{out}},L_{\max}).
\]

对每条路径 \(p\) 与每个输入分级 \((s_1,s_2)\in\{0,1\}^2\)，有可学习通道混合权：
\[
W_{p}^{(s_1,s_2)}\in\mathbb R^{C_{\text{out}}\times C_1\times C_2}.
\]
> 备注（Implementation）：
> 对固定路径 \(p\)，实现一般对四种输入分级组合 \((s_1,s_2)\in\{0,1\}^2\) 分别参数化权重 \(W_{p}^{(s_1,s_2)}\)（即“四套权重”）。

### 5.2 输出分级规则
定义
\[
s_{\text{out}} = s_1\oplus s_2\oplus \mathbf 1_{\text{use\_epsilon}},
\]
其中 \(\oplus\) 是 XOR。其含义为：
- true × true 经 δ 仍为 true
- ε 引入 \(\det R\)，翻转 true/pseudo
- pseudo 的 \((\det R)\) 因子通过 XOR 规则在输出端传递

> 备注（Implementation）：
> 分级规则在代码中常以异或实现（例如 `s_out = s1 ^ s2 ^ (1 if use_epsilon else 0)`），与式 (5.2) 的数学含义一致。

### 5.3 分块形式（形式化定义）
记输入块
\[
X_{L_1}^{(s_1)}\in \mathbb R^{C_1}\otimes\mathcal T_{L_1},\qquad
Y_{L_2}^{(s_2)}\in \mathbb R^{C_2}\otimes\mathcal T_{L_2}.
\]
对每条路径 \(p\)（确定 \(L_1,L_2,L_{\text{out}}\)）定义中间张量
\[
U_{p}^{(s_1,s_2)} :=
\Gamma_p\!\left(X_{L_1}^{(s_1)},\,Y_{L_2}^{(s_2)}\right)
\in (\mathbb R^{C_1}\otimes\mathbb R^{C_2})\otimes\mathcal T_{L_{\text{out}}}.
\]
然后进行通道混合：
\[
Z_{L_{\text{out}}}^{(s_{\text{out}})}
\;+=\;
\sum_{a=1}^{C_1}\sum_{b=1}^{C_2}
W_{p}^{(s_1,s_2)}[\cdot,a,b]\;\,U_{p,a,b}^{(s_1,s_2)}.
\]
最后对所有 \(p,s_1,s_2\) 求和得到完整输出 \(Z\in\mathcal V(C_{\text{out}},L_{\max})\)。

### 5.4 定理：严格 \(O(3)\) 等变性
为书写紧凑，先将第 2 节的分级作用记为表示：对秩 \(L\) 与分级 \(s\in\{0,1\}\)，定义
\[
\rho_{L,s}(R):\mathcal T_L\to\mathcal T_L,\qquad
\rho_{L,s}(R)T:=(\det R)^s\,(R\cdot T).
\tag{5.1}
\]
对通道扩张 \(\mathbb R^C\otimes\mathcal T_L\) 取 \(\mathrm{id}_{\mathbb R^C}\otimes\rho_{L,s}(R)\)，并在直和空间 \(\mathcal V(C,L_{\max})\) 上按分块作用。

### Theorem 1（张量积算子 \(\mathrm{TP}\) 的严格 \(O(3)\) 等变性）
设 \(\mathrm{TP}\) 按第 5.3 节由路径集合与权重 \(W_{p}^{(s_1,s_2)}\) 定义，且输出分级满足
\[
s_{\text{out}} = s_1\oplus s_2\oplus \epsilon,
\qquad \epsilon=\mathbf 1_{\text{use\_epsilon}}.
\tag{5.2}
\]
则对任意 \(R\in O(3)\) 与任意输入 \(x_1\in\mathcal V(C_1,L_{\max}),x_2\in\mathcal V(C_2,L_{\max})\)，有
\[
\mathrm{TP}(R\cdot x_1,\ R\cdot x_2)=R\cdot \mathrm{TP}(x_1,x_2),
\tag{5.3}
\]
其中左右两侧的 \(R\cdot(\,\cdot\,)\) 均指按式 (5.1) 的分级群作用。

**Proof.** 由于 \(\mathcal V\) 为有限直和空间，且 \(\mathrm{TP}\) 可写为对各 rank/parity 块及各路径的有限线性叠加，故仅需验证每个固定块与固定路径对应的单项贡献满足 \(O(3)\) 等变性。

固定输入分级 \((s_1,s_2)\) 与路径 \(p=(k,\epsilon)\)，令 \(\Gamma_p=\Gamma_{k,\epsilon}\)。由 Proposition 1，对任意 \(R\in O(3)\)：
\[
\Gamma_p(R\cdot A,\ R\cdot B)=(\det R)^{\epsilon}\, \big(R\cdot\Gamma_p(A,B)\big),
\tag{5.4}
\]
其中右侧的 \(R\cdot\) 是自然张量作用（不含 \((\det R)^s\)）。于是对分级作用有
\[
\begin{aligned}
\Gamma_p\big(\rho_{L_1,s_1}(R)A,\ \rho_{L_2,s_2}(R)B\big)
&=\Gamma_p\big((\det R)^{s_1}(R\cdot A),\ (\det R)^{s_2}(R\cdot B)\big)\\
&=(\det R)^{s_1+s_2}\ \Gamma_p(R\cdot A,\ R\cdot B)\\
&=(\det R)^{s_1+s_2+\epsilon}\ \big(R\cdot\Gamma_p(A,B)\big)\\
&=(\det R)^{s_{\text{out}}}\ \big(R\cdot\Gamma_p(A,B)\big)\\
&=\rho_{L_{\text{out}},s_{\text{out}}}(R)\,\Gamma_p(A,B),
\end{aligned}
\tag{5.5}
\]
其中第四步使用分级规则 (5.2)（注意 \(s_1\oplus s_2\oplus \epsilon\) 等价于模 2 加法），最后一步是 \(\rho\) 的定义。故几何收缩 \(\Gamma_p\) 在分级空间上严格 \(O(3)\) 等变。

通道混合权重 \(W_{p}^{(s_1,s_2)}\) 仅作用于通道维 \(\mathbb R^{C_1}\otimes\mathbb R^{C_2}\)（其在群作用下为平凡表示），因此与 \(\rho_{L_{\text{out}},s_{\text{out}}}(R)\) 交换；最后，有限求和保持等变性，得到式 (5.3)。□

---

## 6. Pure-Cartesian-Sparse（严格 O(3)）张量积：`PureCartesianTensorProductO3Sparse`

`PureCartesianTensorProductO3Sparse` 与第 5 节相同，仅将路径集合从全枚举裁剪为稀疏子集 \(\mathcal P_{\text{sparse}}\)，仍严格 \(O(3)\) 等变。

### 6.1 稀疏路径集合（实现 `_enumerate_paths_sparse`）
设 `max_rank_other`（默认 1）。仅保留满足
\[
\min(L_1,L_2)\le \texttt{max\_rank\_other}
\]
的交互：至少一侧的秩不超过 `max_rank_other`，每次交互至少有一个低秩张量（标量/向量；`max_rank_other=2` 时允许二阶张量）以减少计算。

### 6.2 `k_policy`（仅针对向量-张量）
当 \(\min(L_1,L_2)=1\) 时，δ 收缩允许的 \(k\) 由
- `"k0"`：只保留 \(k=0\Rightarrow L_{\text{out}}=L_1+L_2\)（向量外积提升秩）
- `"k1"`：只保留 \(k=1\Rightarrow L_{\text{out}}=L_1+L_2-2\)（一次 δ 收缩降低秩）
- `"both"`：两者都保留
控制。

### 6.3 `allow_epsilon`
若 `allow_epsilon=False`，则完全删除 ε 路径；此时本层只用 δ，不会产生新的 pseudo 分量。

### 6.4 `share_parity_weights`
- `share_parity_weights=False`：每个 \((s_1,s_2)\) 一套 \(W_{p}^{(s_1,s_2)}\)（最多 4 套）。
- `share_parity_weights=True`：仅按 \(s_{\text{out}}\in\{0,1\}\) 共享两套权重 \(W_p^{(0)},W_p^{(1)}\)。
这只是参数共享约束，不改变 \(\Gamma_p\) 与分级规则，故不影响等变性。

### 6.5 `assume_pseudo_zero` + `allow_epsilon=False`
当 `assume_pseudo_zero=True` 且 `allow_epsilon=False` 时，仅计算 true×true→true（\(s_1=s_2=s_{\text{out}}=0\)），并省略 pseudo 通道及其对应权重项。该算子仍严格 \(O(3)\) 等变：其仅包含 \(\delta\) 收缩，而 \(\Gamma_p\) 在 \(O(3)\) 下严格等变且不引入 \(\det R\) 因子；输出限制在 true 子空间同样保持等变性。

---

## 7. 边几何张量：\(n^{\otimes L}\)（与实现一致）

对边向量 \(r_{ij}\in\mathbb R^3\)，定义单位方向
\[
n_{ij}=\frac{r_{ij}}{\|r_{ij}\|}.
\]
构造秩 \(L\) 的几何张量（实现 `edge_rank_powers`）：
\[
E_L(n_{ij}) := n_{ij}^{\otimes L}\in\mathcal T_L,
\quad L=0,1,\dots,L_{\max}.
\]
它在 \(O(3)\) 下按自然张量表示变换（且本身是 true 张量，不额外带 \(\det R\)）。

网络中的消息传递用这些 \(E_L\) 与节点标量通道相乘得到 rank 特征，再通过上述 TP 做双线性混合。

---

## 8. 小结（实现到数学的一一对应）

- **表示**：\(\bigoplus_{s\in\{0,1\}}\bigoplus_{L\le L_{\max}}\mathbb R^C\otimes(\mathbb R^3)^{\otimes L}\)
- **群作用**：\((\det R)^s\prod R\) 逐指标作用
- **路径**：\(\delta\) 成对收缩（\(k\) 次）与可选一次 ε 收缩
- **分级传播**：\(s_{\text{out}}=s_1\oplus s_2\oplus \mathbf 1_{\varepsilon}\)
- **稀疏**：只裁剪路径集合与共享权重结构，等变性保持不变

---

## 9. Irreps 表示与 Pure-Cartesian 张量积的对应关系

本节阐述工程实践中常用的 irreps（如 `64x0e + 64x1o + 64x2e`）与 pure-cartesian（\(3^L\)）的对应关系。

### 9.1 `64x0e + 64x1o + 64x2e` 的块结构

这是 irreps 的直和，表示一个特征向量按 \((\ell,p)\) 分块：
- \(64\times 0e\)：64 个**标量**通道（\(\ell=0\)，even parity）
- \(64\times 1o\)：64 个**极向量**通道（\(\ell=1\)，odd parity），每个通道 3 个分量
- \(64\times 2e\)：64 个**二阶不可约张量**通道（\(\ell=2\)，even parity），每个通道 5 个分量（STF 基）

为使“\(e/o\)”成为可推导的数学对象，这里给出 \(O(3)\) 不可约块 \((\ell,p)\) 的标准定义（并与第 11.0 节“宇称扩展”的表述一致）。

### Definition 1（\(O(3)\) 不可约表示块及其宇称标签）
令 \(\ell\in\mathbb N\)，并令 \(D^{(\ell)}:SO(3)\to \mathrm{GL}(2\ell+1)\) 表示 \(SO(3)\) 的 \(\ell\) 阶不可约表示。对任意 \(R\in O(3)\)，存在唯一分解
\[
R = (-I)^{k}\,\widetilde R,\qquad k\in\{0,1\},\ \widetilde R:=(-I)^k R\in SO(3).
\tag{9.1}
\]
对任意 \(p\in\{+1,-1\}\)，定义 \(O(3)\) 的表示
\[
D^{(\ell,p)}(R):=p^{\,k}\,D^{(\ell)}(\widetilde R).
\tag{9.2}
\]
则 \(D^{(\ell,p)}\) 限制到 \(SO(3)\) 时等于 \(D^{(\ell)}\)，且满足 \(D^{(\ell,p)}(-I)=p\,I_{2\ell+1}\)。这正是 e3nn 中 `e/o` 的数学含义：`e` 对应 \(p=+1\)，`o` 对应 \(p=-1\)。

### Proposition 2（true/pseudo 分级与 \(e/o\) 宇称标签的对应关系）
对 pure-cartesian 的块 \((L,s)\)（第 2 节），取全反演 \(R=-I\) 得
\[
(-I)\cdot T^{(s)} = (-1)^{L+s}\,T^{(s)}.
\tag{9.3}
\]
因此该块的“宇称”与 irreps 记号中的 \(p\) 一致地满足 \(p=(-1)^{L+s}\)。特别地，true（\(s=0\)）与 pseudo（\(s=1\)）的差别等价于把宇称标签翻转 \(p\mapsto -p\)。

> **STF（Symmetric Trace-Free，对称无迹）**：STF 张量是满足以下两个条件的张量：
> 1. **对称性**：\(T_{ij} = T_{ji}\)（对 rank-2 张量）
> 2. **无迹性**：\(\mathrm{tr}(T) = \sum_{i} T_{ii} = 0\)（所有迹为零）
> 
> 对于 rank-2 张量，全 \(3\times 3\) 矩阵有 9 个独立分量。对称性约束减少到 6 个分量（\(T_{ij} = T_{ji}\)），无迹性再减少 1 个约束（\(\sum_i T_{ii} = 0\)），因此 STF 子空间维度为 \(6-1=5\)，对应 \(SO(3)\) 的 \(\ell=2\) 不可约表示（维度 \(2\ell+1=5\)）。
> 
> 数学上，任意对称 rank-2 张量 \(T_{ij}\) 可以唯一分解为：
> \[
> T_{ij} = \underbrace{\Big(\tfrac12(T_{ij}+T_{ji})-\tfrac13\delta_{ij}T_{kk}\Big)}_{\text{STF 部分（}\ell=2\text{，5 分量）}} + \underbrace{\tfrac13\delta_{ij}T_{kk}}_{\text{迹部分（}\ell=0\text{，1 分量）}}
> \]
> STF 部分在旋转下按 \(\ell=2\) 不可约表示变换，是 irreps 表示中 \(\ell=2\) 块的基础。

总维度为
\[
64(2\cdot0+1)+64(2\cdot1+1)+64(2\cdot2+1)=64(1+3+5)=576.
\]

把一个节点特征写为
\[
x = \big(x_{0e}, x_{1o}, x_{2e}\big),
\]
则形状对应为（以 batch 维省略）：
- \(x_{0e}\in\mathbb R^{64}\)
- \(x_{1o}\in\mathbb R^{64}\otimes \mathbb R^{3}\)
- \(x_{2e}\in\mathbb R^{64}\otimes \mathbb R^{5}\)

\(\mathbb R^{3}\) 与 \(\mathbb R^{5}\) 并非任意 3 或 5 维向量，而是 **O(3)/SO(3) 的不可约表示空间**（带固定的基与 \(D^\ell(R)\) 作用）。

### 9.2 Irreps 张量积：Clebsch–Gordan 耦合与通道混合

在 irreps 表示框架下，张量积对每条允许耦合路径 \((\ell_1,p_1)\otimes(\ell_2,p_2)\to(\ell_{\text{out}},p_{\text{out}})\) 用 Clebsch–Gordan（CG）系数 [6,8,14] 将 \(m\) 指标耦合成 \(\ell_{\text{out}}\) 的分量，再做通道混合。选择规则：

- **角动量三角条件**：
\[
|\ell_1-\ell_2|\le \ell_{\text{out}}\le \ell_1+\ell_2.
\]
- **宇称（parity）相乘**：
\[
p_{\text{out}} = p_1\,p_2,\qquad p\in\{+1,-1\},\ e\equiv +1,\ o\equiv -1.
\]

对固定的 \((\ell_1,\ell_2,\ell_{\text{out}})\)，耦合的坐标形式可写成
\[
z_{\text{out},m}
 = \sum_{m_1,m_2} C^{\ell_{\text{out}},m}_{\ell_1,m_1;\ \ell_2,m_2}\;
x_{\ell_1,m_1}\;y_{\ell_2,m_2},
\]
其中 \(C\) 是 Clebsch–Gordan（CG）系数 [6,8,14]。在采用 Condon–Shortley 相位约定时，它可由 Wigner-3j 符号 [7,8,15] 表达为：
\[
C^{\ell_3,m_3}_{\ell_1,m_1;\ \ell_2,m_2}
=(-1)^{\ell_1-\ell_2+m_3}\sqrt{2\ell_3+1}
\begin{pmatrix}
\ell_1 & \ell_2 & \ell_3\\
m_1 & m_2 & -m_3
\end{pmatrix}.
\]
再对通道 \((a,b)\) 到输出通道 \(c\) 乘以权重并求和（与 `EquivariantTensorProduct` 中的 `W[c,a,b]` 一致）。

示例：\(\ell=1\) 与 \(\ell=1\) 的乘积允许
\[
1\otimes 1 \to 0\oplus 1\oplus 2,
\]
而宇称会决定这些输出是 \(0e/1e/2e\) 还是 \(0o/1o/2o\)（取决于输入的 \(p_1p_2\)）。

### 9.3 Pure-Cartesian 张量表示：\(3^L\) 全秩张量空间

pure-cartesian 改用
\[
\mathcal T_L=(\mathbb R^3)^{\otimes L}\quad(\dim=3^L)
\]
作为秩 \(L\) 的表示空间。对于 `channels=64, Lmax=2`，其块为：
- rank 0：\(\mathbb R^{64}\otimes \mathcal T_0 \cong \mathbb R^{64}\)（1 分量）
- rank 1：\(\mathbb R^{64}\otimes \mathcal T_1 \cong \mathbb R^{64}\otimes\mathbb R^3\)（3 分量）
- rank 2：\(\mathbb R^{64}\otimes \mathcal T_2 \cong \mathbb R^{64}\otimes(\mathbb R^3\otimes\mathbb R^3)\)（9 分量）

与 irreps 的关键差异在于：rank 2 在 pure-cartesian 中为全 \(3\times 3\)（9 分量），而 irreps 的 \(\ell=2\) 是其中的 **对称无迹（STF）** 子空间（5 分量）。

### 9.4 pure-cartesian 的张量积：δ/ε 外积与收缩

pure-cartesian 的基本双线性算子由路径 \(p\) 定义（第 4 节）：
- **外积**：两个张量拼接，秩相加
- **δ 收缩**：用 \(\delta_{ij}\) 成对消去左右各一个指标（秩减少 2）
- **ε 收缩**：用 \(\varepsilon_{ijk}\) 消去左右各一个指标，产生新指标（秩减少 1），引入 \(\det R\)

> 备注（Implementation）：
> 实现可将路径 \(p=(k,\epsilon)\) 对应的指标收缩写成 Einstein 求和并调用张量库执行；这等价于第 4 节的坐标定义 (4.1)–(4.2)。

### 9.5 pure-cartesian 需要 true/pseudo 的原因

由于 \(\varepsilon\) 在 \(O(3)\) 下满足 \(R^{\otimes 3}\varepsilon = (\det R)\varepsilon\)，ε 路径会多出 \(\det R\)。为使输出在 \(O(3)\) 下按表示空间闭合，需将特征空间分为 true/pseudo 两类，用 \(s_{\text{out}} = s_1\oplus s_2\oplus \mathbf 1_{\varepsilon}\) 追踪 \(\det R\) 因子。

### 9.6 总结：两种张量操作定义的对照

- irreps（例如 `64x0e+64x1o+64x2e`）：
  - 张量积 = CG（Wigner-3j）耦合 \(m\) 指标 + 通道混合
  - \(\ell\) 是不可约阶数，块维度是 \(2\ell+1\)
  - \(\ell=2\) 是 STF（5 分量）
- pure-cartesian：
  - 张量积 = δ/ε 的指标外积与收缩（路径求和）+ 通道混合
  - rank \(L\) 块维度是 \(3^L\)
  - rank 2 是全 \(3\times 3\)（9 分量），不会先投影到 STF

---

## 10. 数学基础：对称张量、齐次多项式与调和多项式

本节阐述 ICTD 方法的数学基础（对称张量、调和多项式与 trace-chain/Fischer 分解），它们用于在 Cartesian 坐标系下实现不可约表示（irreps）相关的投影与耦合（见第 11 节）。

### 10.1 对称 rank-\(L\) 张量 \(\leftrightarrow\) 三元齐次多项式（次数 \(L\)）的同构

令 \(V:=\mathbb R^3\)。记 \(\mathrm{Sym}^L(V)\) 为对称 \(L\) 次张量空间（等价地，\(V\) 上的对称 \(L\)-线性形式），\(\mathcal P_L\) 为三元齐次多项式（次数 \(L\)）空间。

### Definition 2（对称张量空间与齐次多项式空间之间的同构 \(\Phi\)）
定义线性映射
\[
\Phi:\mathrm{Sym}^L(V)\to\mathcal P_L,\qquad
\Phi(T)(x):=T(x,\ldots,x),\ \ x\in V.
\tag{10.1}
\]
在标准基下，若 \(T\) 的坐标为 \(T_{i_1\cdots i_L}\)（对称于指标置换），则
\[
\Phi(T)(x)=\sum_{i_1,\ldots,i_L=1}^3 T_{i_1\cdots i_L}\,x_{i_1}\cdots x_{i_L},
\tag{10.2}
\]
与本文此前记号一致。

### Definition 3（逆映射 \(\Psi\) 的定义：极化公式与偏导表达）
给定 \(p\in\mathcal P_L\)，定义 \(\Psi(p)\in\mathrm{Sym}^L(V)\) 为
\[
\Psi(p)(u_1,\ldots,u_L)
:=\frac{1}{L!}\frac{\partial^L}{\partial t_1\cdots \partial t_L}\,
p\!\left(\sum_{r=1}^L t_r u_r\right)\Bigg|_{t_1=\cdots=t_L=0},
\tag{10.3}
\]
其中 \(u_1,\ldots,u_L\in V\) 任意。该式给出一个对称 \(L\)-线性形式，因此 \(\Psi(p)\in\mathrm{Sym}^L(V)\)。

### Proposition 3（映射 \(\Phi\) 的线性同构性）
\(\Phi\) 与 \(\Psi\) 互为逆映射，从而 \(\mathrm{Sym}^L(V)\cong\mathcal P_L\) 线性同构。

**Proof.** 先说明 \(\Psi\) 的良定义：对固定 \(p\in\mathcal P_L\)，映射
\[
(u_1,\ldots,u_L)\ \longmapsto\ p\!\left(\sum_{r=1}^L t_r u_r\right)
\]
对每个 \(u_r\) 线性、对 \((t_1,\ldots,t_L)\) 是次数 \(L\) 的多项式；因此右端对 \(t_1,\ldots,t_L\) 的混合 \(L\) 阶偏导在 \(t=0\) 处存在，并且对 \(u_1,\ldots,u_L\) 是多线性的。由于混合偏导 \(\partial_{t_1}\cdots\partial_{t_L}\) 在变量置换下不变，\(\Psi(p)\) 还是对称的 \(L\)-线性形式，即 \(\Psi(p)\in\mathrm{Sym}^L(V)\)。

接着证明 \(\Psi\circ\Phi=\mathrm{id}\)。取任意 \(T\in\mathrm{Sym}^L(V)\)，令 \(p=\Phi(T)\)，则
\[
p\!\left(\sum_{r=1}^L t_r u_r\right)
=T\!\left(\sum_{r=1}^L t_r u_r,\ldots,\sum_{r=1}^L t_r u_r\right).
\]
将右端按多线性展开后，项 \(\prod_{r=1}^L t_r\) 的系数恰为 \(L!\,T(u_1,\ldots,u_L)\)（来自 \(L\) 个位置上分别选取 \(u_1,\ldots,u_L\) 的所有排列；对称性保证所有排列给出同一数值）。因此
\[
\frac{1}{L!}\frac{\partial^L}{\partial t_1\cdots\partial t_L}\,
p\!\left(\sum_{r=1}^L t_r u_r\right)\Bigg|_{t=0}
=T(u_1,\ldots,u_L),
\]
即 \(\Psi(\Phi(T))=T\)。

最后证明 \(\Phi\circ\Psi=\mathrm{id}\)。取任意 \(p\in\mathcal P_L\)，令 \(T=\Psi(p)\)。对任意 \(x\in V\)，取 \(u_1=\cdots=u_L=x\) 并令 \(t_1=\cdots=t_L=t\)，则
\[
p\!\left(\sum_{r=1}^L t_r u_r\right)=p(Ltx)=L^L t^L p(x),
\]
其中最后一步使用 \(p\) 的 \(L\) 次齐次性。另一方面，由 \(\Psi\) 的定义与对称性，左端对 \(t\) 的 \(L\) 阶导数在 \(t=0\) 处与 \(T(x,\ldots,x)\) 成正比；直接比较 \(t^L\) 的系数可得 \(T(x,\ldots,x)=p(x)\)，即 \(\Phi(T)=p\)。因此 \(\Phi(\Psi(p))=p\)。证毕。

**维度校验**。有
\[
\dim\mathrm{Sym}^L(\mathbb R^3)=\binom{L+2}{2}=\dim\mathcal P_L,
\tag{10.4}
\]
与同构结论一致。

**关于 \(O(3)\) 作用（后续将用）**。在多项式侧，定义 \(O(3)\) 的自然作用（pullback）
\[
(U_L(R)p)(x):=p(R^\top x),\qquad R\in O(3).
\tag{10.5}
\]
它满足 \(U_L(R_1R_2)=U_L(R_1)U_L(R_2)\)，因而给出 \(\mathcal P_L\) 上的表示；并且在同构 \(\Phi\) 下与对称张量的自然作用一致。

> 备注（实现中使用的单项式系数坐标）：
> 任意 \(p\in\mathcal P_L\) 可写作 \(p(x,y,z)=\sum_{a+b+c=L} t_{abc}x^ay^bz^c\)。系数 \(t_{abc}\) 与张量坐标 \(T_{i_1\cdots i_L}\) 的对应关系是 \(\Phi\) 在该基下的坐标表达；其中组合权 \(w_{abc}=L!/(a!\,b!\,c!)\) 正是“计数为 \((a,b,c)\)”的指标轨道大小。

### 10.2 STF（对称无迹）\(\Longleftrightarrow\) 调和多项式（\(\Delta p=0\)）

对称张量的“迹”定义为用 \(\delta_{ij}\) 收缩两条指标得到的对称张量：
\[
(\mathrm{tr}\,T)_{i_3\cdots i_L}:=\sum_{j=1}^3 T_{jj\,i_3\cdots i_L}\in \mathrm{Sym}^{L-2}(V).
\tag{10.6}
\]
称 \(T\in\mathrm{Sym}^L(V)\) 为 **STF（symmetric trace-free）**，若 \(\mathrm{tr}\,T=0\)（等价地，所有迭代迹均为零）。

在多项式侧，定义 Laplacian
\[
\Delta := \partial_x^2+\partial_y^2+\partial_z^2:\mathcal P_L\to\mathcal P_{L-2}.
\tag{10.7}
\]

### Proposition 4（迹算子与 Laplacian 的对应关系）
令 \(T\in\mathrm{Sym}^L(V)\)，并记 \(p_T=\Phi(T)\)。则
\[
\Delta p_T = L(L-1)\, p_{\mathrm{tr}(T)}.
\tag{10.8}
\]

**Proof.** 由式 (10.2) 展开并计算二阶偏导：
\[
\partial_{x_a}\partial_{x_a}\left(x_{i_1}\cdots x_{i_L}\right)
 = \sum_{1\le r\ne s\le L} \delta_{a i_r}\delta_{a i_s}\ x_{i_1}\cdots \widehat{x_{i_r}}\cdots \widehat{x_{i_s}}\cdots x_{i_L},
\]
将其与对称性合并计数可得系数 \(L(L-1)\)，并恰好产生 \(T_{aa\,i_3\cdots i_L}\) 的收缩，即式 (10.8)。□

### Corollary 1（STF 条件与调和条件的等价性）
定义调和子空间
\[
\mathcal H_L:=\ker(\Delta)\subset\mathcal P_L.
\tag{10.9}
\]
则 \(\Phi\) 将 STF 子空间 \(\ker(\mathrm{tr})\subset\mathrm{Sym}^L(V)\) 线性同构到 \(\mathcal H_L\)。特别地，
\[
\mathrm{tr}(T)=0\ \Longleftrightarrow\ \Delta p_T=0.
\tag{10.10}
\]

**维度与分解（Fischer / trace-chain）**。记 \(r^2:=x^2+y^2+z^2\)。经典的 Fischer 分解在三维给出直和分解
\[
\mathcal P_L=\mathcal H_L\ \oplus\ r^2\,\mathcal P_{L-2}.
\tag{10.11}
\]
递归应用得到
\[
\mathcal P_L=\bigoplus_{k=0}^{\lfloor L/2\rfloor} r^{2k}\mathcal H_{L-2k}.
\tag{10.12}
\]
由维数计算 \(\dim\mathcal P_L=\binom{L+2}{2}\) 与 \(\dim(r^2\mathcal P_{L-2})=\binom{L}{2}\)，得到
\[
\dim\mathcal H_L=\binom{L+2}{2}-\binom{L}{2}=2L+1,
\tag{10.13}
\]
这正是三维 \(SO(3)\) 的 \(\ell=L\) 不可约表示维数。

> 备注：式 (10.11)–(10.12) 是第 11 节 trace-chain 投影 \(P_{L\to\ell}\) 的数学基础；实现中的“trace-chain”即对应从 \(\mathcal P_L\) 中依次剥离 \(r^{2k}\mathcal H_{L-2k}\) 的分量。

---

## 11. ICTD-irreps（trace-chain irreps 内部表示，**无球谐**）的描述

> 备注（Implementation）：
> 本节的数学对象对应于工程实现中的 ICTD-irreps 路线：网络内部特征以 \((2\ell+1)\) 的 irreps 块表示，并通过“多项式乘法 + trace-chain 投影”构造耦合张量。

### 11.0 本节与 O(3)（宇称/反射）的关系

本节核心对象是 \(\mathcal H_\ell\)（调和多项式 / STF）与 trace-chain（Fischer）分解。它们最常见的表述是在 \(SO(3)\) 下作为 \(\ell\) 阶不可约表示（维度 \(2\ell+1\)）。然而，在多项式模型中更自然的做法是直接在 \(O(3)\) 上定义群作用（第 10.1 节式 (10.5)）：
\[
(U_\ell(R)p)(x):=p(R^\top x),\qquad R\in O(3).
\tag{11.1}
\]
由于该作用对所有 \(R\in O(3)\) 都良定义，\(\mathcal H_\ell=\ker\Delta\) 自动成为 \(O(3)\) 的不变子空间（见 11.0.1），从而给出一个 \(O(3)\) 表示。

需要强调的是：\(SO(3)\) 的 \(\ell\) 阶不可约表示在扩展到 \(O(3)\) 时存在两种“宇称类型”（相差一个一维符号表示）。在多项式模型下，\(\mathcal H_\ell\) 诱导得到的是其中的 **自然宇称** 扩展：对全反演 \(-I\) 满足 \(U_\ell(-I)=(-1)^\ell I\)（见 11.0.1）。若要得到另一种（det-扭曲的）宇称类型，可与符号表示 \(\mathrm{sgn}(R)=\det R\) 作张量积，从而将所有 \(\det R\) 因子作为显式标签在层间传播（与第 2 节 true/pseudo 的记号一致）。

> 备注（实现层面的结论）：
> `ictd_irreps.py` / `pure_cartesian_ictd_layers.py` 的单元测试一般以随机旋转 \(R\in SO(3)\) 验证严格等变；若将测试扩展到一般 \(R\in O(3)\)，则需同时明确采用上述两类宇称扩展中的哪一类（以及 \(\det R\) 标签在通道中的显式携带约定）。

#### 11.0.1 ICTD/调和多项式构造在 \(O(3)\)（含宇称）下的等变性

核心是：ICTD 的全部运算（Laplacian、乘以 \(r^2\)、以及基/投影的构造）都与 \(O(3)\) 的 pullback 作用 \(U_L(R)\) 可交换，因此得到的投影与耦合都是 \(O(3)\) 的交织算子。

### Lemma 2（\(\Delta\) 与 \(U_L\) 的交换性）
对任意 \(R\in O(3)\) 与任意 \(p\in\mathcal P_L\)，有
\[
\Delta(U_L(R)p)=U_{L-2}(R)(\Delta p).
\tag{11.2}
\]
因此 \(\mathcal H_L=\ker\Delta\) 对任意 \(R\in O(3)\) 都是不变子空间。

### Lemma 3（乘以 \(r^{2k}\) 与 \(U\) 的交换性）
对任意 \(R\in O(3)\)，有 \(\|R^\top x\|^2=\|x\|^2\)，故对任意 \(k\ge 0\)：
\[
U_L(R)\big(r^{2k}q\big)=r^{2k}\,U_{L-2k}(R)q.
\tag{11.3}
\]

### Lemma 4（各向同性高斯内积的 \(O(3)\) 不变性）
定义
\[
\langle p,q\rangle_G:=\mathbb E_{X\sim\mathcal N(0,I_3)}[p(X)q(X)].
\tag{11.4}
\]
则对任意 \(R\in O(3)\)，有 \(\langle U_L(R)p,\ U_L(R)q\rangle_G=\langle p,q\rangle_G\)。因此由该内积诱导的 Gram 矩阵与任意 \(U_L(R)\) 对易。

由引理 4 可知：若用 \(\langle\cdot,\cdot\rangle_G\) 在 \(\mathcal P_L\) 中做最小二乘/正交投影，则该投影算子与 \(U_L(R)\) 交换，从而是 \(O(3)\)-intertwiner。结合 Fischer 分解（第 10.2 节式 (10.12)），从 \(\mathcal P_L\) 投影到某个 \(\mathcal H_\ell\) 分量的 trace-chain 投影 \(P_{L\to\ell}\) 因此与 \(U(R)\) 交换。

### Corollary 2（多项式乘法与 trace-chain 投影所诱导的 \(O(3)\) 交织算子）
多项式乘法映射
\[
m:\mathcal H_{\ell_1}\times\mathcal H_{\ell_2}\to\mathcal P_{\ell_1+\ell_2},\qquad (p,q)\mapsto pq
\tag{11.5}
\]
满足 \(m(U_{\ell_1}(R)p,\ U_{\ell_2}(R)q)=U_{\ell_1+\ell_2}(R)\,m(p,q)\)。因此复合 \(P_{\ell_1+\ell_2\to \ell_3}\circ m\) 是 \(O(3)\) 的交织算子，对应三阶耦合张量 \(C^{\ell_3}_{\ell_1\ell_2}\)。

**宇称（自然宇称）**。对全反演 \(R=-I\)，对任意齐次次数为 \(\ell\) 的多项式 \(p\) 有
\[
(U_\ell(-I)p)(x)=p(-x)=(-1)^\ell p(x),
\tag{11.6}
\]
因此 \(\mathcal H_\ell\) 在该 \(O(3)\) 扩展下天然携带 \((-1)^\ell\) 的宇称。这与球谐函数的性质一致。

### 11.1 记号

- 令 \(V=\mathbb{R}^3\)，\(R\in O(3)\)。
- \(\mathrm{Sym}^L(V)\)：对称 rank-\(L\) 张量空间，维度 \(\binom{L+2}{2}\)。
- \(\mathcal{P}_L\)：三元齐次多项式（次数 \(L\)）空间，与 \(\mathrm{Sym}^L(V)\) 同构。
- \(\Delta=\partial_x^2+\partial_y^2+\partial_z^2\)：Laplacian（把次数 \(L\) 降到 \(L-2\)）。
- \(\mathcal{H}_L:=\ker(\Delta)\subset \mathcal{P}_L\)：调和（harmonic）子空间，\(\dim\mathcal{H}_L=2L+1\)。它对应 irreps 的 \(\ell=L\) 块（STF）。
- \(r^2=x^2+y^2+z^2\)。

### 11.2 对称张量 \(\leftrightarrow\) 齐次多项式

本节使用与 **10.1** 相同的同构：对称 rank-\(L\) 张量 \(\mathrm{Sym}^L(\mathbb R^3)\) 与次数 \(L\) 的三元齐次多项式空间 \(\mathcal P_L\) 线性等价（见第 10.1 节）。

取单项式基 \(m_{abc}(x)=x^a y^b z^c\)（\(a+b+c=L\)）。任意 \(t\in\mathrm{Sym}^L(V)\)（在该基下的系数向量记为 \(t_L\)）对应多项式
\[
p_t(x)=\sum_{a+b+c=L} (t_L)_{abc}\, x^a y^b z^c.
\]
旋转 \(R\) 对多项式的自然作用是变量替换
\[
(U_L(R)p)(x)=p(R^T x),
\]
它在 \(\mathcal{P}_L\) 上诱导出一个表示（也是 \(\mathrm{Sym}^L(V)\) 的表示）。

### 11.3 Harmonic（STF）子空间：\(\ker(\Delta)\)

\[
\mathcal{H}_L:=\{p\in\mathcal{P}_L:\Delta p=0\}.
\]
它是"对称无迹（STF）"子空间的多项式版本，维度为 \(2L+1\)。关于 STF 与调和多项式的对应关系，见第 10.2 节。

> 备注（Implementation）：
> 在具体实现中，一般在某个显式系数基（例如单项式系数基）下构造 \(\Delta:\mathcal P_L\to\mathcal P_{L-2}\) 的矩阵表示，再通过零空间计算得到 \(\mathcal H_L\) 的一组基向量。

### 11.4 用 **\(O(3)\) 不变**的高斯 Gram 固定基（核心步骤）

本节旨在 \(\mathcal H_L\) 上选取一个与 \(O(3)\) 作用相容的正交归一基，从而使诱导表示矩阵 \(D^{(L)}(R)\) 的定义具有一致性：不同基的选择至多相差一个正交基变换（见命题 5）。

### Definition 4（各向同性高斯内积）
在 \(\mathcal P_L\) 上定义内积
\[
\langle p,q\rangle_G := \mathbb{E}_{X\sim\mathcal{N}(0,I_3)}[p(X)q(X)].
\]
等价地（忽略一个与归一化无关的常数因子）：
\[
\langle p,q\rangle_G \ \propto\ \int_{\mathbb R^3} p(x)\,q(x)\,e^{-\|x\|^2/2}\,dx.
\]
它对 \(O(3)\) 不变，因此对 \(SO(3)\) 也不变（引理 4）。

在单项式基下，该内积对应 Gram 矩阵 \(G_L\)，其元素为
\[
(G_L)_{(abc),(a'b'c')}=
\mathbb{E}[x^{a+a'}]\ \mathbb{E}[y^{b+b'}]\ \mathbb{E}[z^{c+c'}],
\]
其中一维高斯矩
\[
\mathbb{E}[x^n]=
\begin{cases}
0,& n\ \text{为奇数}\\
(n-1)!!,& n\ \text{为偶数}.
\end{cases}
\]

### Proposition 5（\(\mathcal H_L\) 的 \(G\)-正交归一基：存在性与唯一性）
设 \(B_L\in\mathbb{R}^{D_L\times(2L+1)}\) 的列向量张成 \(\mathcal H_L\)（\(D_L=\binom{L+2}{2}\)）。则存在可逆矩阵 \(A\in\mathbb R^{(2L+1)\times(2L+1)}\) 使得 \(\widetilde B_L:=B_LA\) 满足
\[
\widetilde B_L^T G_L \widetilde B_L = I_{2L+1}.
\]
并且若 \(\widehat B_L\) 也是 \(\mathcal H_L\) 的另一组 \(G\)-正交归一基（满足 \(\widehat B_L^T G_L\widehat B_L=I\)），则存在正交矩阵 \(Q\in O(2L+1)\) 使得 \(\widehat B_L=\widetilde B_LQ\)。

**Proof (sketch).** \(\langle\cdot,\cdot\rangle_G\) 在 \(\mathcal P_L\) 上正定，限制到子空间 \(\mathcal H_L\) 仍正定，因此 \(B_L^T G_L B_L\) 对称正定；取其 Cholesky 分解（或谱分解）得到白化矩阵 \(A\)。唯一性来自：同一内积下两组正交归一基之间的过渡矩阵必为正交矩阵。□

> 备注（Implementation）：
> 命题 5 阐明了采用各向同性内积进行“正交化”的依据：任意零空间基在数值上可能非常不稳定，而 \(G\)-正交化得到的基在 \(O(3)\) 不变结构下更一致；同时，基的非唯一性被严格限定为右乘一个正交矩阵。

### 11.5 trace-chain 分解与投影 \(P_{L\to \ell}\)

经典分解（harmonic decomposition / trace chain）：
\[
\mathcal{P}_L=\bigoplus_{k=0}^{\lfloor L/2\rfloor} r^{2k}\mathcal{H}_{L-2k}.
\]
给定 \(\ell=L-2k\)，定义提升算子（工程实现中常用矩阵表示该线性映射）：
\[
M_{\ell,k}:\mathcal{H}_\ell\to \mathcal{P}_L,\quad h\mapsto r^{2k}h.
\]
\(V_{\ell,k}:=M_{\ell,k}B_\ell\) 是子空间 \(r^{2k}\mathcal H_\ell\subset\mathcal P_L\) 的一组列基。

### Definition 5（trace-chain 分量的 \(G_L\)-正交投影与块坐标）
对任意 \(t_L\in\mathbb R^{D_L}\)（表示 \(\mathcal P_L\) 的系数向量），定义其 \(\ell\) 块坐标 \(c_\ell\in\mathbb R^{2\ell+1}\) 为满足：\(V_{\ell,k}c_\ell\) 是 \(t_L\) 在内积 \(\langle\cdot,\cdot\rangle_G\) 下到子空间 \(r^{2k}\mathcal H_\ell\) 的正交投影。

当 \(V_{\ell,k}\) 满列秩时（其列向量线性无关），该投影唯一且正规方程给出：
\[
c_\ell = P_{L\to \ell}\, t_L,
\quad
P_{L\to \ell}:=(V_{\ell,k}^T G_L V_{\ell,k})^{-1}V_{\ell,k}^T G_L.
\]
若 \(V_{\ell,k}\) 不满列秩，则可用 Moore–Penrose 伪逆替代 \((V_{\ell,k}^T G_L V_{\ell,k})^{-1}\) 给出最小范数解；本文后续默认满列秩情形（在理论上由 Fischer 分解保证子空间维数为 \(2\ell+1\)，因此只要 \(B_\ell\) 取为基且 \(M_{\ell,k}\) 在该子空间上单射便可满足）。

\(c_\ell\in\mathbb{R}^{2\ell+1}\) 为 irreps 块坐标（在固定的 harmonic 基下）。

### 11.6 无球谐的边方向特征 \(Y_\ell(n)\)

给定单位方向向量 \(n\in\mathbb{R}^3\)，定义齐次多项式
\[
p_{n,\ell}(x)=(n\cdot x)^\ell\in\mathcal{P}_\ell.
\]
把它展开为单项式系数向量 \(t_\ell(n)\)，再投影到 harmonic 块得到
\[
Y_\ell(n):=B_\ell^T G_\ell\, t_\ell(n)\in\mathbb{R}^{2\ell+1}.
\]
这等价于把 \(p_{n,\ell}\) 在 \(\langle\cdot,\cdot\rangle_G\) 下投影到 \(\mathcal H_\ell\)，并取其在 \(G\)-正交归一基 \(B_\ell\) 下的坐标。

### Proposition 6（特征 \(Y_\ell\) 的等变性）
在第 11.0 节的 \(O(3)\) 作用下，
\[
Y_\ell(Rn)=D^{(\ell)}(R)\,Y_\ell(n),
\]
其中可取
\[
D^{(\ell)}(R):=B_\ell^T G_\ell\,U_\ell(R)\,B_\ell\in\mathbb R^{(2\ell+1)\times(2\ell+1)}.
\tag{11.8}
\]

**Proof (sketch).** 由 Lemma 4，\(U_\ell(R)\) 为 \(\langle\cdot,\cdot\rangle_G\) 的等距变换，从而保持 \(\mathcal H_\ell\) 并在任意 \(G\)-正交归一基下对应一个正交矩阵 \(D^{(\ell)}(R)\)。此外，\(p_{Rn,\ell}(x)=(Rn\cdot x)^\ell=(n\cdot R^T x)^\ell=(U_\ell(R)p_{n,\ell})(x)\)。综合两式即得。□

**归一化/基的一致性说明**：此处构造的 \(Y_\ell(n)\) 是在本实现选择的 harmonic 基 \(B_\ell\) 与内积 \(G_\ell\) 下的坐标。它与常见的（实）球谐函数基 \(Y_{\ell m}(\hat n)\) 一般不要求逐分量相等，但二者承载同一个 \((2\ell+1)\) 维 \(SO(3)\) 不可约表示，至多相差一个 \((2\ell+1)\times(2\ell+1)\) 的正交变换；变换性质完全一致：\(Y_\ell(Rn)=D^{(\ell)}(R)Y_\ell(n)\)。

### 11.7 在这套基下构造 CG（耦合张量）而不使用 Wigner-3j

给定 \(a\in\mathcal{H}_{\ell_1}\)、\(b\in\mathcal{H}_{\ell_2}\)（在基下坐标分别为 \(a_{m_1}, b_{m_2}\)），考虑多项式乘积：
\[
p_a(x)\,p_b(x)\in\mathcal{P}_{L},\quad L=\ell_1+\ell_2.
\]
再用 trace-chain 投影提取 \(\ell_3\) 块：
\[
c_{\ell_3}:=P_{L\to \ell_3}\big(p_a p_b\big)\in\mathcal{H}_{\ell_3}.
\]
这等价于存在一个三阶耦合张量（CG 张量）
\[
C^{\ell_3}_{\ell_1\ell_2}\in\mathbb{R}^{(2\ell_1+1)\times(2\ell_2+1)\times(2\ell_3+1)}
\]
使得
\[
(c_{\ell_3})_{m_3}=\sum_{m_1,m_2} a_{m_1}b_{m_2}\ (C^{\ell_3}_{\ell_1\ell_2})_{m_1m_2m_3}.
\]
构造自动满足选择规则：

- 三角不等式 \(|\ell_1-\ell_2|\le \ell_3\le \ell_1+\ell_2\)；
- \(\ell_1+\ell_2-\ell_3\) 必须为偶数（来自 \(r^{2k}\) 的 trace-chain 结构）。

### Proposition 7（Clebsch–Gordan 交织算子的构造：多项式乘法与投影）
定义双线性算子
\[
\mathrm{TP}_{\ell_1,\ell_2\to\ell_3}:\mathcal H_{\ell_1}\times\mathcal H_{\ell_2}\to\mathcal H_{\ell_3},\qquad
\mathrm{TP}_{\ell_1,\ell_2\to\ell_3}(a,b):=P_{\ell_1+\ell_2\to \ell_3}\big(p_a p_b\big),
\tag{11.9}
\]
其中 \(p_a,p_b\) 表示与坐标 \(a,b\) 对应的调和多项式（同构 \(\mathcal H_\ell\subset\mathcal P_\ell\)）。则对任意 \(R\in O(3)\) 有
\[
\mathrm{TP}_{\ell_1,\ell_2\to\ell_3}\big(D^{(\ell_1)}(R)a,\ D^{(\ell_2)}(R)b\big)
=D^{(\ell_3)}(R)\,\mathrm{TP}_{\ell_1,\ell_2\to\ell_3}(a,b),
\tag{11.10}
\]
即 \(\mathrm{TP}_{\ell_1,\ell_2\to\ell_3}\) 是 \(O(3)\)-交织双线性算子（与主文档 `PAPER_CROSS_LAYER_FUSION_THEORY.md` 的 Definition 4 形式一致）。

**Proof (sketch).** 由 Corollary 2，乘法 \(m\) 与投影 \(P\) 均与 \(U(R)\) 交换；将 \(\mathcal H_\ell\) 上的 \(O(3)\) 作用写为 \(D^{(\ell)}(R)\) 的坐标矩阵，即得式 (11.10)。□

由命题 7，\(\mathrm{TP}_{\ell_1,\ell_2\to\ell_3}\) 在任意固定基下的坐标表示必形如本节所述三阶张量 \(C^{\ell_3}_{\ell_1\ell_2}\)。不同基的选择只会对该三阶张量施加独立的正交基变换（命题 5），其交织性质不变；相位约定（如 Condon–Shortley）对应于在该等价类中选取一个具体代表。

**交换对称性（补充）**：由于多项式乘法满足交换律 \(p_a p_b=p_b p_a\)，交换输入 \((\ell_1,m_1)\leftrightarrow(\ell_2,m_2)\) 时，耦合张量对应交换前两维的同一个 intertwiner。采用标准 Condon–Shortley 相位约定的 CG 系数时，常见关系为
\[
(C^{\ell_3}_{\ell_1\ell_2})_{m_1m_2m_3}
=(-1)^{\ell_1+\ell_2-\ell_3}\,(C^{\ell_3}_{\ell_2\ell_1})_{m_2m_1m_3}.
\]
在本实现构造中，具体符号/相位可能随所选基 \(B_\ell\) 发生整体正交变换，但交换输入对应确定的符号/基变换这一结构保持不变。

### 11.8 可学习版本：shared weights + per-path gating

网络内部特征采用 irreps 分块：
\[
x^{(\ell)}\in\mathbb{R}^{\mu\times(2\ell+1)},
\]
其中 \(\mu\) 是该 \(\ell\) 的 multiplicity（通道数）。

对每条边 \(e\)，径向网络输出每条“耦合路径” \(p=(\ell_1,\ell_2\to\ell_3)\) 的标量 gate：
\[
g_e(p)\in\mathbb{R}.
\]
每条路径存一个共享权重张量
\[
W_p\in\mathbb{R}^{\mu_{\text{out}}\times\mu_1\times\mu_2}.
\]
则路径贡献为
\[
m_{e,p}^{(\ell_3)} = g_e(p)\cdot
\Big[\;W_p\cdot_{\mu_1,\mu_2}\big(x^{(\ell_1)}\otimes_{C^{\ell_3}_{\ell_1\ell_2}} y^{(\ell_2)}\big)\;\Big],
\]
对所有 \(p\) 求和得到边消息 \(m_e^{(\ell_3)}\)。由于 \(g_e(p)\) 是标量、\(\otimes_C\) 是交织算子（intertwiner），整体在所选的表示作用下保持等变性；若 \(\otimes_C\) 取自第 11.0.1 节的 \(O(3)\) 交织构造，则该等变性对所有 \(R\in O(3)\) 成立（在浮点实现中将体现为数值误差意义下的等变性）。

---

## 12. 张量积模式性能对比（Benchmark 结果）

本节提供所有六种张量积实现模式的性能对比数据，包括参数量、前向传播速度、反向传播速度和等变性验证结果。

### 12.0 理论等变性与数值等变误差
为避免将“数学保证”与“数值观测”混淆，本文对等变性使用两套互补表述：

1. **理论等变性（intertwiner 性质）**：若某一层/算子 \(F\) 在精确算术下满足
\[
F(\rho_{\mathrm{in}}(R)x)=\rho_{\mathrm{out}}(R)F(x),\qquad \forall R\in G,
\tag{12.1}
\]
则称其对群 \(G\)（本文常取 \(G=O(3)\)）严格等变。第 4–6 节给出 pure-cartesian 情形的严格证明；第 10–11 节给出 ICTD/多项式路线下的交织构造（其在数值实现时需要额外讨论浮点误差）。

2. **数值等变误差（实验指标）**：对给定实现（浮点数）与随机采样的 \(R\sim \mu\)（例如 Haar(SO(3)) 及若干 \(\det(R)=-1\) 的不当正交变换），定义相对误差
\[
\mathrm{err}(F;R,x):=
\frac{\big\|F(\rho_{\mathrm{in}}(R)x)-\rho_{\mathrm{out}}(R)F(x)\big\|}{\big\|F(x)\big\|+\varepsilon},
\tag{12.2}
\]
其中 \(\varepsilon>0\) 为防止分母为零的稳定项。本文报告的“等变性误差”均指对若干随机 \((R,x)\) 的统计量（例如最大值或均值），其数量级反映实现的数值稳定性与浮点舍入误差，而非替代理论证明。

### 12.1 测试环境

#### 12.1.1 CPU 测试环境

- **硬件**：CPU
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：前向传播 30 次平均，反向传播 20 次平均
- **等变性测试**：按式 (12.2) 计算的数值等变误差；群元素采样包含 \(SO(3)\) 随机旋转并额外加入若干 \(\det(R)=-1\) 的不当正交变换（以覆盖 \(O(3)\) 宇称分支）
- **数据类型**：float64
- **注意**：`pure-cartesian` 仅报告到 lmax=3；当 lmax≥4 时在当前硬件/实现下未能完成测试（资源限制）

#### 12.1.2 GPU 测试环境

- **硬件**：NVIDIA GeForce RTX 3090
- **模型配置**：`channels=64`, `lmax=0` 到 `lmax=6`
- **测试数据**：32 atoms, 256 edges
- **测试方法**：前向传播 30 次平均，反向传播 20 次平均
- **等变性测试**：同 CPU（按式 (12.2) 的数值等变误差；覆盖 \(O(3)\) 两个连通分支）
- **数据类型**：float64

### 12.2 性能对比总览（lmax=2，典型配置）

基于 lmax=2 的测试结果（详见 12.6 和 12.7 节）：

| 模式 | 理论等变性（精确算术） | 参数量 (lmax=2) | 相对参数量 | CPU 加速比 | GPU 加速比 | 数值等变误差 |
|------|--------|----------------|------------|------------|------------|------------|
| `spherical` | \(O(3)\) 交织（严格） | 6,540,634 | 100% (基准) | 1.00x (基准) | 1.00x (基准) | \(\sim 10^{-15}\) |
| `partial-cartesian` | \(O(3)\) 交织（严格） | 5,404,938 | 82.6% | 1.06x | 0.75x | \(\sim 10^{-14}\) |
| `partial-cartesian-loose` | 无严格保证（近似构造） | 5,406,026 | 82.7% | 1.33x | 1.15x | \(\sim 10^{-15}\) |
| `pure-cartesian` | \(O(3)\) 交织（严格） | 33,626,186 | 514.0% | 0.06x | 0.06x | \(\sim 10^{-14}\) |
| `pure-cartesian-sparse` | \(O(3)\) 交织（严格） | 4,606,026 | 70.4% | 1.39x | 1.17x | \(\sim 10^{-15}\) |
| `pure-cartesian-ictd` | 交织构造（理论上 \(O(3)\)；实现为数值近似） | 1,824,497 | **27.9%** | **4.12x** | **2.10x** | \(\sim 10^{-7}\) |

**观察到的结论（在本节测试设置下）**：
- `pure-cartesian-ictd` 在所有报告的 lmax 上具有最高的吞吐加速比（CPU 最大约 4.12x；GPU 最大约 2.10x）
- `pure-cartesian-ictd` 的参数量最小（约为 spherical 的 27–32%）
- `pure-cartesian` 在 lmax≥4 时未能完成测试（资源限制），因此未报告对应数据

### 12.3 参数量详细对比（lmax=0 到 6）

参数量随 lmax 的变化趋势（详见 12.6.2 和 12.7.2 节）：

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`partial-cartesian` 参数量减少 17-18%**（82-83% of spherical）
- **`pure-cartesian` 在 lmax≥4 时未完成测试**（资源限制）

### 12.4 等变性验证（O(3)，包含宇称）

等变性误差随 lmax 的变化（详见 12.6.3 和 12.7.3 节）：

**CPU 环境（典型数量级）**：
- `spherical`/`pure-cartesian-sparse`/`pure-cartesian`：数值误差接近机器精度（\(\sim 10^{-15}\) 到 \(10^{-14}\)）
- `pure-cartesian-ictd`：数值误差为 \(10^{-7}\) 到 \(10^{-6}\)（与其数值构造方式/条件数相关）

**GPU 环境（典型数量级）**：
- `spherical`/`pure-cartesian-sparse`：数值误差接近机器精度（\(\sim 10^{-15}\)）
- `pure-cartesian-ictd`：数值误差约为 \(10^{-7}\)（见 12.7.3 表）

> 备注：上表“数值误差”的大小不应被解读为等变性的理论判定。例如，即便理论上为交织算子的实现，也可能由于预计算矩阵的数值条件数、正交化/最小二乘过程的舍入误差等而出现 \(10^{-7}\) 量级的偏差；相反，缺乏理论保证的近似方法亦可能在特定测试设定下呈现较小误差。

### 12.5 性能分析

基于完整测试结果（lmax=0 到 6）的综合分析：

1. **参数量优化**：
   - **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical），在所有 lmax 下都保持优势
   - **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical），平衡了性能和参数量
   - **`partial-cartesian` 参数量减少 17-18%**（82-83% of spherical），略优于基准

2. **速度优化**：
   - **`pure-cartesian-ictd` 在本测试设置下加速比最高**（CPU: 最大约 4.12x@lmax=2；GPU: 最大约 2.10x@lmax=2）
   - **`pure-cartesian-sparse` 表现稳定**：在所有 lmax 下都接近或快于基准（CPU: 0.53x - 1.39x，GPU: 0.46x - 1.17x）
   - **`partial-cartesian-loose` 在低 lmax 时较快**（CPU: 1.13x - 1.37x，GPU: 1.15x - 1.52x）；该模式属于近似构造，一般不提供理论交织保证（其在本实验中的误差大小仅反映该测试设置下的数值现象）

3. **等变性（理论 vs 数值）**：
   - `spherical`/`pure-cartesian`/`pure-cartesian-sparse`：理论上为 \(O(3)\) 交织算子，且在本实验中误差接近机器精度
   - `pure-cartesian-ictd`：按第 10–11 节为交织构造，但在当前数值实现与测试设置下出现 \(10^{-7}\)–\(10^{-6}\) 级误差
   - `partial-cartesian-loose`：无严格保证；在本实验中误差较小但不代表一般情形

4. **实践备注（基于本节实验）**：
   - `pure-cartesian`（非稀疏）在本实验中速度较慢且资源开销较大；并且在 lmax≥4 未完成测试（资源限制）。因此除非研究需要，一般不作为默认配置

### 12.6 CPU 测试结果（lmax=0 到 6）

#### 12.6.1 总训练时间加速比（前向+反向，相对于 spherical）

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             1.06x |                  1.13x |           0.36x |                 1.07x |                **2.97x** |
| 1    |             1.05x |                  1.37x |           0.13x |                 1.02x |                **3.33x** |
| 2    |             1.06x |                  1.33x |           0.06x |                 1.39x |                **4.12x** |
| 3    |             0.58x |                  0.70x |           0.02x |                 1.05x |                **2.68x** |
| 4    |             0.37x |                  0.43x |        **FAILED** |                 0.97x |                **2.20x** |
| 5    |             0.23x |                  0.28x |        **FAILED** |                 0.78x |                **1.81x** |
| 6    |             0.16x |                  0.17x |        **FAILED** |                 0.53x |                **1.58x** |

**性能分析**：
- **`pure-cartesian-ictd` 在本测试设置下加速比最高**（最大约 **4.12x**@lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（2.68x - 4.12x）
- **lmax = 4-6**: 在本测试设置下 `pure-cartesian-ictd` 仍保持最高加速比（1.58x - 2.20x）
- **`pure-cartesian-sparse` 表现稳定**：在所有 lmax 下都接近基准（0.53x - 1.39x）

#### 12.6.2 参数量对比（CPU，lmax=0 到 6）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`pure-cartesian` 在 lmax≥4 时未完成测试**（资源限制）

#### 12.6.3 等变误差对比（CPU，O(3)，包含宇称）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 3.33e-15  |        8.24e-16  |            6.39e-14  |      3.41e-15  |           4.34e-15  |           1.69e-12 |
| 1    | 5.58e-14  |        3.33e-16  |            1.65e-15  |      5.01e-15  |           7.24e-15  |           3.80e-08 |
| 2    | 3.88e-15  |        1.87e-14  |            7.73e-16  |      3.08e-15  |           3.56e-14  |           7.73e-08 |
| 3    | 1.16e-15  |        2.47e-14  |            3.71e-15  |      8.14e-16  |           9.63e-16  |           7.23e-08 |
| 4    | 1.27e-15  |        6.83e-15  |            1.00e-15  |       **FAILED** |           1.24e-14  |           3.50e-05 |
| 5    | 3.43e-15  |        3.45e-15  |            2.27e-15  |       **FAILED** |           3.59e-14  |           8.11e-08 |
| 6    | 3.26e-15  |        2.01e-15  |            5.82e-16  |       **FAILED** |           1.51e-15  |           1.00e-06 |

**性能分析**：
- **数值等变误差接近机器精度**（\(\sim 10^{-15}\)）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
- **数值等变误差显著高于机器精度**（\(\sim 10^{-7}\) 到 \(10^{-6}\)）：`pure-cartesian-ictd`

### 12.7 GPU 测试结果（lmax=0 到 6）

#### 12.7.1 总训练时间加速比（前向+反向，相对于 spherical）

| lmax | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    |             0.96x |                  1.52x |           0.54x |                 1.17x |                **1.92x** |
| 1    |             0.85x |                  1.33x |           0.16x |                 1.02x |                **1.97x** |
| 2    |             0.75x |                  1.15x |           0.06x |                 1.17x |                **2.10x** |
| 3    |             0.56x |                  0.81x |           0.02x |                 1.15x |                **1.91x** |
| 4    |             0.38x |                  0.51x |        **FAILED** |                 0.99x |                **1.78x** |
| 5    |             0.26x |                  0.32x |        **FAILED** |                 0.75x |                **1.44x** |
| 6    |             0.17x |                  0.21x |        **FAILED** |                 0.46x |                **1.05x** |

**性能分析**：
- **`pure-cartesian-ictd` 在本测试设置下加速比最高**（最大约 **2.10x**@lmax=2）
- **lmax ≤ 3**: `pure-cartesian-ictd` 优势最明显（1.91x - 2.10x）
- **lmax = 4-5**: `pure-cartesian-ictd` 仍然更快（1.44x - 1.78x）
- **lmax = 6**: `pure-cartesian-ictd` 和 `spherical` 几乎相等（1.05x）

#### 12.7.2 参数量对比（GPU，lmax=0 到 6）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 2,313,018 |        1,128,074 |            1,128,074 |      2,725,450 |           1,128,010 |             626,653 |
| 1    | 4,164,682 |        2,734,026 |            2,734,434 |     11,786,058 |           2,733,898 |           1,208,931 |
| 2    | 6,540,634 |        5,404,938 |            5,406,026 |     33,626,186 |           4,606,026 |           1,824,497 |
| 3    | 8,654,426 |        9,407,050 |            9,409,090 |     74,635,594 |           6,744,394 |           2,481,673 |
| 4    | 10,768,218 |       15,272,842 |           15,276,106 |         **FAILED** |           8,882,762 |           3,197,103 |
| 5    | 12,882,010 |       23,268,554 |           23,273,314 |         **FAILED** |          11,021,130 |           3,979,109 |
| 6    | 14,995,802 |       33,926,666 |           33,933,194 |         **FAILED** |          13,159,498 |           4,844,335 |

**性能分析**：
- **`pure-cartesian-ictd` 参数量始终最少**（27-32% of spherical）
- **`pure-cartesian-sparse` 参数量适中**（70-88% of spherical）
- **`pure-cartesian` 在 lmax≥4 时未完成测试**（资源限制）

#### 12.7.3 等变误差对比（GPU，O(3)，包含宇称）

| lmax | spherical | partial-cartesian | partial-cartesian-loose | pure-cartesian | pure-cartesian-sparse | pure-cartesian-ictd |
|------|----------:|------------------:|----------------------:|---------------:|---------------------:|-------------------:|
| 0    | 1.03e-15  |        6.37e-16  |            1.20e-15  |      3.27e-15  |           9.31e-16  |           5.24e-08 |
| 2    | 9.27e-16  |        1.97e-16  |            8.96e-16  |      3.18e-14  |           1.30e-15  |           7.18e-07 |
| 4    | 9.16e-16  |        7.76e-14  |            4.26e-16  |       **FAILED** |           6.19e-16  |           7.16e-07 |
| 6    | 4.77e-16  |        7.02e-16  |            5.65e-16  |       **FAILED** |           5.72e-16  |           1.11e-07 |

**性能分析**：
- **数值等变误差接近机器精度**（\(\sim 10^{-15}\)）：`spherical`, `partial-cartesian-loose`, `pure-cartesian-sparse`
- **数值等变误差显著高于机器精度**（\(\sim 10^{-7}\)）：`pure-cartesian-ictd`

### 12.8 推荐使用场景

#### 12.8.1 CPU 环境（基于本节实验的实践建议）
- 若以吞吐/参数效率为主要目标，可优先考虑 `pure-cartesian-ictd`（在本实验中最高约 4.12x 加速，且参数量最少）
- 若以数值等变误差接近机器精度为主要目标，可优先考虑 `spherical` 或 `pure-cartesian-sparse`
- `pure-cartesian` 在高 lmax 下资源开销较大；在本实验中 lmax≥4 未能完成测试，故不作为默认选择

#### 12.8.2 GPU 环境（基于本节实验的实践建议）
- 若以吞吐/参数效率为主要目标，可优先考虑 `pure-cartesian-ictd`（本实验中最高约 2.10x 加速）
- 若以数值等变误差接近机器精度为主要目标，可优先考虑 `spherical` 或 `pure-cartesian-sparse`
- `pure-cartesian` 在高 lmax 下资源开销较大；本实验中 lmax≥4 未能完成测试

### 12.9 实际任务测试结果

**数据集**：五条氮氧化物和碳结构反应路径的 NEB（Nudged Elastic Band）数据，截取到 fmax=0.2，总共 2,788 条数据。测试集：每个反应选取 1-2 条完整或不完整的数据。

**测试配置**：64 channels, lmax=2, float64

#### 12.9.1 能量和力精度对比

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
<td rowspan="5" style="text-align:center;vertical-align:middle"><strong>FSCETP</strong></td>
<td rowspan="5" style="text-align:center;vertical-align:middle">Lmax=2, 64ch</td>
<td style="text-align:center">spherical</td>
<td style="text-align:center">0.044 ⭐</td>
<td style="text-align:center">7.4</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian</td>
<td style="text-align:center">0.045</td>
<td style="text-align:center">7.4</td>
</tr>
<tr>
<td style="text-align:center">partial-cartesian-loose</td>
<td style="text-align:center">0.048</td>
<td style="text-align:center">8.4</td>
</tr>
<tr>
<td style="text-align:center"><strong>pure-cartesian-sparse</td>
<td style="text-align:center"><strong>0.044 ⭐</strong></td>
<td style="text-align:center"><strong>6.5 ⭐</strong></td>
</tr>
<tr>
<td style="text-align:center">pure-cartesian-ictd</td>
<td style="text-align:center">0.046</td>
<td style="text-align:center">9.0</td>
</tr>
</tbody>
</table>

**说明**：
- 数值越小越好
- ⭐ 表示该指标的最佳结果

#### 12.9.2 性能分析

**能量精度对比**：
- **FSCETP 相比 MACE (64ch) 的能量 RMSE 降低了 66.2%**
  - FSCETP 最优结果：0.044 mev/atom（`spherical` 与 `pure-cartesian-sparse` 模式并列）
  - MACE (64ch) 基准值：0.13 mev/atom
  - 相对误差比：0.34（FSCETP / MACE）

**力精度对比**：
- **FSCETP 相比 MACE (64ch) 的力 RMSE 降低了 44.0%**
  - FSCETP 最优结果：6.5 mev/Å（`pure-cartesian-sparse` 模式）
  - MACE (64ch) 基准值：11.6 mev/Å
  - 相对误差比：0.56（FSCETP / MACE）

**模式性能总结**：
在该数据集与该测试配置（lmax=2, 64ch, float64）下的观察结论：
- **能量 RMSE 最低（并列）**：`spherical` 与 `pure-cartesian-sparse`（0.044 mev/atom）
- **力 RMSE 最低**：`pure-cartesian-sparse`（6.5 mev/Å）
- **效率与精度的折中**：`pure-cartesian-ictd` 在能量/力误差接近最优的同时，具有更小参数量与更高吞吐（具体数值见本节表格；其“折中”结论仅针对本数据集与本配置）

**结论**：
- 所有 FSCETP 模式在真实化学反应路径数据集上均显著优于 MACE（64ch、128ch 和 198ch 配置）
- `pure-cartesian-sparse` 在该测试配置下取得最优力 RMSE，并与 `spherical` 并列最优能量 RMSE
- `pure-cartesian-ictd` 模式在保持竞争性精度的同时，提供了显著的参数效率和速度优势

### 12.10 数学实现对应关系

| 模式 | 内部表示 | 张量积方法 | 理论等变性（精确算术） |
|------|----------|------------|------------|
| `spherical` | irreps `64x0e+64x1o+64x2e` | Wigner-3j/CG 耦合 | \(O(3)\) 交织（严格） |
| `partial-cartesian` | irreps `64x0e+64x1o+64x2e` | 笛卡尔坐标 + CG 系数 | \(O(3)\) 交织（严格） |
| `partial-cartesian-loose` | irreps `64x0e+64x1o+64x2e` | 近似（non-intertwiner） | 无严格保证 |
| `pure-cartesian` | \(\bigoplus_{L\le 2}\mathbb R^{64}\otimes(\mathbb R^3)^{\otimes L}\) | δ/ε 路径枚举 | \(O(3)\) 交织（严格；见第 4–5 节） |
| `pure-cartesian-sparse` | \(\bigoplus_{L\le 2}\mathbb R^{64}\otimes(\mathbb R^3)^{\otimes L}\) | δ/ε 路径稀疏化 | \(O(3)\) 交织（严格） |
| `pure-cartesian-ictd` | irreps `64x0e+64x1o+64x2e` | trace-chain 投影 + 多项式耦合 | 交织构造（理论上 \(O(3)\)；实现可能受数值误差影响） |

---

## 参考文献

[1] Geiger, M., & Smidt, T. (2022). e3nn: Euclidean neural networks. *arXiv preprint arXiv:2207.09453*.

[2] Zee, A. (2016). *Group theory in a nutshell for physicists*. Princeton University Press.

[3] Shao, S., Li, Y., Lin, Z., & Cui, Q. (2025). High-rank irreducible Cartesian tensor decomposition and bases of equivariant spaces. *Journal of Machine Learning Research*, 26(175), 1-53. arXiv:2412.18263. (注：该工作通过 path matrices 直接构造不可约表示；本实现则通过调和多项式的 Laplacian 零空间方法构建不可约表示，见第 11 节)

[4] Weiler, M., Geiger, M., Welling, M., Boomsma, W., & Cohen, T. (2018). 3D steerable CNNs: Learning rotationally equivariant features in volumetric data. In *Advances in Neural Information Processing Systems* (Vol. 31).

[5] Thomas, N., Smidt, T., Kearnes, S., Yang, L., Li, L., Kohlhoff, K., & Riley, P. (2018). Tensor field networks: Rotation-and translation-equivariant neural networks for 3D point clouds. *arXiv preprint arXiv:1802.08219*.

[6] Biedenharn, L. C., & Louck, J. D. (1981). *Angular momentum in quantum physics: theory and application*. Cambridge University Press.

[7] Wigner, E. P. (1959). *Group theory and its application to the quantum mechanics of atomic spectra*. Academic Press.

[8] Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988). *Quantum theory of angular momentum*. World Scientific.

[9] Stone, A. J. (2013). *The theory of intermolecular forces* (2nd ed.). Oxford University Press.

[10] Kondor, R., & Trivedi, S. (2018). On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *Proceedings of the 35th International Conference on Machine Learning* (pp. 2747-2756). PMLR.

[11] Cohen, T. S., Geiger, M., Köhler, J., & Welling, M. (2018). Spherical CNNs. In *International Conference on Learning Representations*.

[12] Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., & Kozinsky, B. (2022). E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. *Nature Communications*, 13(1), 2453.

[13] Musaelian, A., Batzner, S., Johansson, A., Sun, L., Owen, C. J., Kornbluth, M., & Kozinsky, B. (2023). Learning local equivariant representations for large-scale atomistic dynamics. *Nature Communications*, 14(1), 579.

[14] Edmonds, A. R. (1957). *Angular momentum in quantum mechanics*. Princeton University Press.

[15] Messiah, A. (1961). *Quantum mechanics* (Vol. 2). North-Holland Publishing Company.
