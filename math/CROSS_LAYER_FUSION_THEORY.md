# 跨层融合（Cross-layer Fusion）的数学分析

> 本文给出跨层融合（cross-layer fusion）结构的形式化分析。在固定的模型截断配置与资源约束（例如角动量截断、通道数与张量积路径集合受限、以及有限深度）下，本文比较不同等变势能模型在读出端构造特征的方式，并给出跨层双线性等变特征（及其不变量化读出）对可生成标量特征子空间与体阶可达性的理论刻画。

---

## 1. 预备定义与假设（Preliminaries）

### Definition 1（局部环境）
固定中心原子 \(i\)。其截断邻域记为
\[
\mathcal{N}(i)=\{j\neq i:\|r_{ij}\|\le r_c\},\qquad r_{ij}=x_j-x_i+\text{PBC shift}.
\]

### Definition 2（对称性作用）
令 \(G=\mathrm{SO}(3)\)。旋转 \(R\in G\) 作用为 \(r_{ij}\mapsto Rr_{ij}\)。邻域置换群 \(S_{\mathcal{N}(i)}\) 作用为索引重排 \(j\mapsto \pi(j)\)。

局部能量函数 \(E_i\) 满足：
\[
E_i(\{Rr_{ij}\}_{j})=E_i(\{r_{ij}\}_j),\qquad
E_i(\{r_{i\pi(j)}\}_j)=E_i(\{r_{ij}\}_j).
\]

### Definition 3（等变特征空间）
对 \(l=0,\dots,l_{\max}\)，记 \(\mathcal{H}_l\) 为维度 \(2l+1\) 的不可约表示空间，表示矩阵为 \(D^{(l)}(R)\)。
第 \(t\) 层在中心 \(i\) 的等变特征属于
\[
h_i^{(t)}\in \bigoplus_{l=0}^{l_{\max}}\left(\mathbb{R}^{c_l^{(t)}}\otimes \mathcal{H}_l\right),
\]
其中 \(c_l^{(t)}\) 表示第 \(t\) 层在角动量 \(l\) 的通道数（multiplicity）。并满足等变性：
\[
h_{i,l}^{(t)}(\{Rr_{ij}\}_j)=(I_{c_l^{(t)}}\otimes D^{(l)}(R))\,h_{i,l}^{(t)}(\{r_{ij}\}_j),
\]
其中 \(I_{c_l^{(t)}}\) 表示 \(c_l^{(t)} \times c_l^{(t)}\) 的单位矩阵。

> 注：e3nn 的不可约表示（irreps）实现、partial Cartesian（strict/loose）以及 ICTD 都可在上述抽象框架中表述；它们描述的是**同一类等变表示空间**在不同坐标系/实现策略下的落地方式。更具体地：
> - **e3nn/irreps**：以不可约表示分解为主，张量积由 intertwiner（Clebsch–Gordan 系数）在 irreps 分量上实现。
> - **partial Cartesian（strict/loose）**：以 Cartesian 坐标组织张量，但核心耦合仍依赖 e3nn 的 irreps 与 Clebsch–Gordan 系数；在实现上对应“在 Cartesian 坐标系下对部分通道/路径实现（可能带截断/近似的）张量积与投影”，因此应理解为 **partial Cartesian** 而非完整的笛卡尔张量积体系。
> - **ICTD**：以调和多项式（harmonic polynomials）/调和张量为基础在 Cartesian 坐标系下构建张量积（可理解为对 irreps 张量积的一种坐标实现），属于“笛卡尔张量积的实现方式”之一。
> - **pure-Cartesian**：以 rank-\(\ell\) Cartesian 张量（坐标维度 \(3^{\ell}\)）为主，通过 δ/ε 收缩与 Gram 型 contraction 实现耦合/不变量化。
> 为避免与“网络深度 \(L\)”混淆，本文用 \(\ell\) 表示张量阶（rank）。

> **附录指引（与本文同一论文）**：pure-Cartesian 及 ICTD 的严格数学定义、\(O(3)\)（含反射/宇称）下的等变性证明与数值等变误差的区分，见附录文档 `PURE_CARTESIAN_MATH.md`。该附录历史上用 \(L\) 表示 Cartesian rank（即主文档的 \(\ell\)），读者可按附录开头的“记号对照表”做一一替换。

### Definition 4（等变双线性张量积与不变量算子）
记 \(V\) 为输入等变特征所在的表示空间，群表示记为 \(D_V(R)\)；记 \(W\) 为输出表示空间，群表示记为 \(D_W(R)\)。称算子 \(\mathrm{TP}(\cdot,\cdot):V\times V\to W\) 为**等变双线性**，若
\[
\mathrm{TP}(D_V(R)u,\ D_V(R)v)=D_W(R)\,\mathrm{TP}(u,v),\quad
\mathrm{TP}(\alpha u_1+\beta u_2,v)=\alpha \mathrm{TP}(u_1,v)+\beta \mathrm{TP}(u_2,v),
\]
并对第二自变量同理。

进一步，称线性算子 \(\mathrm{Inv}:W\to\mathbb{R}^m\) 为**不变量化算子**，若
\[
\mathrm{Inv}(D_W(R)w)=\mathrm{Inv}(w),\qquad \forall R\in G.
\]
在给定 \(\mathrm{TP}\) 与 \(\mathrm{Inv}\) 的情况下，定义
\[
\mathcal{I}(u,v):=\mathrm{Inv}(\mathrm{TP}(u,v))\in \mathbb{R}^m.
\]
当 \(m=1\) 时，\(\mathcal{I}(u,v)\) 退化为标量的**不变量双线性型**，并满足
\[
\mathcal{I}(D(R)u,\ D(R)v)=\mathcal{I}(u,v),\qquad \forall R\in G.
\]

在不可约表示（irreps）坐标系下，\(\mathrm{TP}\) 可由等变张量积（intertwiner / tensor product）实现（其分量耦合由 Clebsch–Gordan 系数刻画），\(\mathrm{Inv}\) 可由投影到平凡表示或其它 \(G\)-不变线性读出实现；在 pure-Cartesian 坐标系下，\(\mathrm{TP}\) 与 \(\mathrm{Inv}\) 可由 δ/ε 收缩或 Gram 型 contraction 实现；在 ICTD（基于调和多项式/调和张量的 Cartesian 实现）下，\(\mathrm{TP}\) 可通过调和基与笛卡尔张量积/投影的组合实现，并在需要时再通过收缩或线性不变量化算子得到 \(\mathrm{Inv}\) 的输出。

**假设 A1（实现层面的可分解性）**：本文讨论的双线性算子在固定坐标系下可写为有限项坐标乘积的线性组合。更形式化地：对任意线性泛函 \(\ell:W\to\mathbb{R}\)（例如 \(\ell=e_q^\top\circ \mathrm{Inv}\)，对应不变量化输出的第 \(q\) 个坐标），存在有限 \(P<\infty\) 与线性算子 \(A_p,B_p\)，使得对任意向量值特征 \(u,v\) 有
\[
\ell(\mathrm{TP}(u,v))=\sum_{p=1}^{P}\langle A_p u,\ B_p v\rangle.
\]
其中 \(\langle\cdot,\cdot\rangle\) 表示坐标上的点乘（等价于有限项的“坐标乘积后求和”）。该假设对本文涉及的主流实现成立（irreps/Cartesian/ICTD/pure-Cartesian）：对任意固定的输出坐标或不变量化坐标，\(\ell(\mathrm{TP}(u,v))\) 都可实现为有限项收缩/点积/Gram 型 contraction 的线性组合，从而满足上述形式。

---

## 2. 多体阶数（Body order）的形式化定义

为避免“对邻居求和是否使体阶 +1”的歧义，我们采用标准的“子集核分解”定义（与 \(N\)-body expansion / Hoeffding-ANOVA 在对称函数空间上的分解一致）。

### Definition 5（体阶 \(\mathrm{bo}(\cdot)\)）
设 \(f\) 是定义在有限多集 \(\{r_{ij}\}_{j\in\mathcal{N}(i)}\) 上的置换不变函数，其取值可为标量或向量（例如等变特征在固定坐标系下的坐标表示）。以下先对**标量情形**给出定义；对向量值 \(f=(f_1,\dots,f_d)\) 的情形，定义 \(\mathrm{bo}(f):=\max_q \mathrm{bo}(f_q)\)，其中每个 \(f_q\) 按本定义计算体阶。称标量 \(f\) 的体阶不超过 \(K\)，若存在一族核函数 \(\{f_S\}\)（对所有 \(S\subseteq \mathcal{N}(i)\) 且 \(|S|\le K\)），使得
\[
f(\{r_{ij}\}_{j\in\mathcal{N}(i)})
=\sum_{\substack{S\subseteq \mathcal{N}(i)\\ |S|\le K}} f_S(\{r_{ij}\}_{j\in S}),
\]
其中每个 \(f_S\) 仅依赖于子集 \(S\) 中的相对位移（及其原子种类等局部属性）。
定义
\[
\mathrm{bo}(f)=\min\{K:\ f \text{ 满足上述表示}\}.
\]

> 注：此定义把“体阶”与“依赖于多少个邻居索引”的概念严格对齐；并与 ACE/MTP/MACE 文献中的 \(N\)-body 子空间一致。

### Definition 5'（\(K\)-body 子空间与正交分解：Hoeffding/ANOVA 视角）
为给出体阶“下界/可达性”的可检验判别，考虑固定最大邻居数 \(n:=|\mathcal{N}(i)|\) 的情形。设 \(\mathcal{X}\) 表示单个相对位移（及元素类型标签等）所在的样本空间，定义
\[
\mathcal{H}:=L^2_{\mathrm{sym}}(\mathcal{X}^n)
\]
为 \(\mathcal{X}^n\) 上置换对称（交换任意邻居坐标不变）的平方可积**标量**函数空间（参考测度可取由数据分布诱导的经验测度或其极限）。对向量值函数，可对每个坐标分量分别应用以下分解与投影。

**假设 A2（Hoeffding-ANOVA 正交分解存在）**：在所选参考测度下，\(\mathcal{H}\) 存在 Hoeffding-ANOVA 的正交分解
\[
\mathcal{H}=\bigoplus_{k=0}^{n}\mathcal{H}^{(k)},
\]
其中 \(\mathcal{H}^{(k)}\) 是“纯 \(k\)-体成分”的闭子空间（与所有低于 \(k\) 体成分正交）。记 \(P_k:\mathcal{H}\to\mathcal{H}^{(k)}\) 为正交投影。
在该框架下，对标量函数体阶可等价刻画为
\[
\mathrm{bo}(f)=\max\{k:\ \|P_k f\|_{L^2}>0\}.
\]
对向量值 \(f=(f_1,\dots,f_d)\)，定义 \(\mathrm{bo}(f):=\max_q \mathrm{bo}(f_q)\)，并对每个 \(f_q\) 分别使用上述投影判别。

> 备注（投影的显式形式）：在（交换）乘积测度设定下，Hoeffding 分量可用条件期望的包含-排除公式给出。令 \(f\in L^2(\mathcal{X}^n)\)，对任意索引子集 \(S\subseteq\{1,\dots,n\}\)，记 \(\mathbb{E}[f\mid S]\) 为在固定坐标 \(x_S\) 条件下、对其余坐标积分的条件期望。定义
> \[
> f_S^{\circ}:=\sum_{U\subseteq S}(-1)^{|S|-|U|}\,\mathbb{E}[f\mid U].
> \]
> 则 \(f=\sum_{S\subseteq[n]} f_S^{\circ}\) 且不同 \(|S|\) 的分量正交；“纯 \(k\)-体子空间”由 \(\{f_S^{\circ}:\ |S|=k\}\) 张成并闭包化得到，从而 \(P_k f=\sum_{|S|=k} f_S^{\circ}\)。

**假设 A3（乘积测度/坐标独立性）**：进一步假设参考测度在坐标上为乘积形式（或至少满足：对互不相交坐标集合 \(A,B\)，在给定任意坐标子集 \(U\) 后，\(A\setminus U\) 与 \(B\setminus U\) 条件独立）。在该假设下，对任意仅依赖于坐标集合 \(A\) 的函数 \(g\) 与仅依赖于坐标集合 \(B\) 的函数 \(h\)，以及任意 \(U\subseteq A\cup B\)，有条件期望的乘积分解
\[
\mathbb{E}[g\,h\mid U]=\mathbb{E}[g\mid U\cap A]\;\mathbb{E}[h\mid U\cap B].
\]

> **注（假设 A3 的适用范围）**：A3 是额外的统计独立性假设，并不由分子体系的几何约束与相互作用结构所蕴含。本文在 A2–A3 下给出体阶分解与可达性的可检验充分条件；当参考测度不满足 A3 时，本文不对体阶可达性给出进一步结论。

> **补充（A3 的物理直观与近似意义）**：在真实分子体系中，原子位置与局部几何往往存在显著相关性，因此坐标分布通常不满足严格的乘积测度假设。尽管如此，在“条件独立近似”或“弱相关”的情形下，Hoeffding-ANOVA 的正交分解仍可作为体阶分析的有效框架：它提供了可操作的体阶上界/下界判别与构造工具，用于解释不同结构在固定深度与有限截断配置下的可达性差异。经验上，基于该框架推导的结构性预测与实际性能提升相一致，提示该近似在实践中具有合理性与解释力。

---

## 3. 基本引理（Lemmas）

### Lemma 1（求和闭包：置换对称求和不增体阶）
若对每个 \(j\in \mathcal{N}(i)\)，函数 \(g_j\) 满足 \(\mathrm{bo}(g_j)\le K\) 且 \(g_j\) 仅依赖于某个至多 \(K\) 个邻居组成的子集（其中可以包含索引 \(j\)），则
\[
g(\{r_{ij}\}_j):=\sum_{j\in\mathcal{N}(i)} g_j(\{r_{ik}\}_{k\in \mathcal{N}(i)})
\]
满足 \(\mathrm{bo}(g)\le K\)。

**Proof.** 由 Definition 5，对每个 \(g_j\) 存在核分解 \(g_j=\sum_{|S|\le K} (g_j)_S\)。则
\[
g=\sum_j g_j=\sum_j\sum_{|S|\le K}(g_j)_S
\]
仍是 \(|S|\le K\) 的核之和，因此 \(\mathrm{bo}(g)\le K\)。□

### Lemma 2（双线性映射导致体阶上界相加）
设 \(u,v\) 为两个（可向量值的）置换对称特征，其每个坐标分量都满足 Definition 5 的核分解，且 \(\mathrm{bo}(u)\le K_u\)、\(\mathrm{bo}(v)\le K_v\)。
令 \(\mathcal{T}:V\times V\to W\) 为任意双线性映射（等变/不变与否不影响体阶结论）。对 \(W\)-值输出，我们定义其体阶为
\[
\mathrm{bo}(\mathcal{T}(u,v)):=\max_{q}\mathrm{bo}\big([\mathcal{T}(u,v)]_q\big),
\]
其中 \([\cdot]_q\) 表示在任意固定坐标系下的第 \(q\) 个坐标分量（由于线性坐标变换不会引入新的索引依赖，该定义与坐标选择相容）。则
\[
\mathrm{bo}(\mathcal{T}(u,v))\le K_u+K_v.
\]

**Proof.** 写 \(u=\sum_{|S|\le K_u}u_S,\ v=\sum_{|T|\le K_v}v_T\) 为核分解。双线性给出
\[
\mathcal{T}(u,v)=\sum_{|S|\le K_u}\sum_{|T|\le K_v}\mathcal{T}(u_S,v_T).
\]
对任意坐标分量 \([\mathcal{T}(u,v)]_q\)，其仍为双线性标量表达的有限和，因此每一项仅依赖于索引并集 \(S\cup T\)，其大小 \(|S\cup T|\le |S|+|T|\le K_u+K_v\)。因此每个分量都存在 \(|U|\le K_u+K_v\) 的核分解，从而 \(\mathrm{bo}(\mathcal{T}(u,v))\le K_u+K_v\)。□

### Lemma 3（体阶下界判别：非零 \(K\)-body 投影蕴含体阶下界）
在 Definition 5' 的 Hilbert 空间设定下，若对某个 \(K\) 有 \(\|P_K f\|_{L^2}>0\)，则 \(\mathrm{bo}(f)\ge K\)。

**Proof.** 由 Definition 5' 的等价刻画，\(\mathrm{bo}(f)=\max\{k:\|P_k f\|_{L^2}>0\}\)。因此 \(\|P_K f\|_{L^2}>0\Rightarrow \mathrm{bo}(f)\ge K\)。□

### Lemma 4（纯成分的等价判别：条件期望为零 \(\Leftrightarrow\) Hoeffding 纯成分）
在假设 A2–A3 的设定下，取任意非空索引集合 \(S\subseteq\{1,\dots,n\}\)。设 \(g\in L^2(\mathcal{X}^n)\) 仅依赖于坐标集合 \(S\)（即 \(g(x)=g(x_S)\)）。则以下两条等价：

1. \(g\in \mathcal{H}^{(|S|)}\) 且其支持集合为 \(S\)（即 \(g\) 为关于 \(S\) 的纯 \(|S|\)-体 Hoeffding 分量）。
2. 对所有真子集 \(U\subsetneq S\)，有 \(\mathbb{E}[g\mid U]=0\)。

并且在上述条件成立时，\(P_{|S|}g=g\)。

**Proof.**
（\(2\Rightarrow 1\)）由于 \(g\) 仅依赖于 \(S\)，其 Hoeffding 分解只可能包含 \(\{U\subseteq S\}\) 上的分量。由 Definition 5' 的显式公式（包含-排除），对任意 \(U\subseteq S\) 有
\[
g_U^{\circ}=\sum_{V\subseteq U}(-1)^{|U|-|V|}\,\mathbb{E}[g\mid V].
\]
当 \(U\subsetneq S\) 时，右端仅涉及 \(V\subsetneq S\)，由条件 \( \mathbb{E}[g\mid V]=0 \) 可知 \(g_U^{\circ}=0\)。而当 \(U=S\) 时，
\[
g_S^{\circ}=\sum_{V\subseteq S}(-1)^{|S|-|V|}\,\mathbb{E}[g\mid V]
=\mathbb{E}[g\mid S]+\sum_{V\subsetneq S}(-1)^{|S|-|V|}\,\mathbb{E}[g\mid V]
=g.
\]
因此 \(g=\sum_{U\subseteq S} g_U^{\circ}=g_S^{\circ}\in \mathcal{H}^{(|S|)}\)，并且 \(P_{|S|}g=g\)。

（\(1\Rightarrow 2\)）若 \(g\in \mathcal{H}^{(|S|)}\) 且仅依赖于 \(S\)，则其 Hoeffding 分解为 \(g=g_S^{\circ}\)。对任意 \(U\subsetneq S\)，由 \(g_S^{\circ}\) 的构造可知其满足“退化性”（degeneracy）：
\[
\mathbb{E}[g_S^{\circ}\mid U]=0,
\]
这等价于其与所有低阶子空间正交（A2），从而 \(\mathbb{E}[g\mid U]=0\)。□

### Lemma 5（Hoeffding-ANOVA：互不相交变量上的“纯成分”乘积）
在 Definition 5' 的设定下，设 \(S,T\subseteq\{1,\dots,n\}\) 且 \(S\cap T=\varnothing\)。令
\[
u\in \mathcal{H}^{(|S|)} \text{ 且 } u \text{ 仅依赖于坐标 } S,\qquad
v\in \mathcal{H}^{(|T|)} \text{ 且 } v \text{ 仅依赖于坐标 } T.
\]
则乘积 \(w:=uv\) 属于 \(\mathcal{H}^{(|S|+|T|)}\) 且仅依赖于 \(S\cup T\)，并满足
\[
P_{|S|+|T|}w=w.
\]

**Proof.** 由 Lemma 4，对所有真子集 \(S'\subsetneq S\) 有 \(\mathbb{E}[u\mid S']=0\)，并且对所有真子集 \(T'\subsetneq T\) 有 \(\mathbb{E}[v\mid T']=0\)。

考虑 \(w=uv\)。显然 \(w\) 仅依赖于 \(S\cup T\)。取任意真子集 \(U\subsetneq S\cup T\)。由假设 A3（乘积测度/坐标独立性），有条件期望的乘积分解
\[
\mathbb{E}[w\mid U]=\mathbb{E}[u\,v\mid U]
=\mathbb{E}[u\mid U\cap S]\;\mathbb{E}[v\mid U\cap T].
\]
由于 \(U\subsetneq S\cup T\)，必有 \(U\cap S\subsetneq S\) 或 \(U\cap T\subsetneq T\)（或二者皆真）。若 \(U\cap S\subsetneq S\)，则 \(\mathbb{E}[u\mid U\cap S]=0\)；若 \(U\cap T\subsetneq T\)，则 \(\mathbb{E}[v\mid U\cap T]=0\)。从而 \(\mathbb{E}[w\mid U]=0\)。
因此 \(\mathbb{E}[w\mid U]=0\) 对所有真子集 \(U\subsetneq S\cup T\) 成立，这正是 \(w\) 属于纯 \(|S|+|T|\) 体成分的判别条件。于是 \(w\in\mathcal{H}^{(|S|+|T|)}\)，并且由于 \(w\) 已经在该子空间内，正交投影满足 \(P_{|S|+|T|}w=w\)。□

### Lemma 6（函数类包含：特征拼接不降低表示能力）
设基线模型输出 \(E=\rho(\psi)\)。扩展模型输出 \(E'=\rho'(\psi\oplus \tilde\psi)\)，其中 \(\rho'\) 与 \(\rho\) 同属足够表达的 MLP 类。则存在 \(\rho'\) 的参数使得 \(E'\equiv E\)（令 \(\rho'\) 忽略 \(\tilde\psi\)）。因此扩展模型的函数类包含基线模型的函数类。□

---

## 4. 跨层融合的主结果（Main Results）

### Definition 6（跨层融合特征：两层/任意 \(L\) 层）
本节区分两类“跨层融合”输出：其一为**等变的跨层融合特征**（输出可为任意不可约表示的直和，类型与维度均为超参数）；其二为上述等变特征的**不变读出/不变量化**（用于能量等标量目标），后者是前者在读出端施加不变量化算子的特例。

（两层版本：等变为主，不变量化为可选读出）设两层等变特征 \(h^{(a)}\in V_a,\,h^{(b)}\in V_b\)（\(a<b\)）。令
\[
u:=h^{(a)}\oplus h^{(b)}\in V_a\oplus V_b,
\]
并选定等变双线性算子
\[
\mathrm{TP}^{\oplus}_{a,b}:(V_a\oplus V_b)\times (V_a\oplus V_b)\to W_{\mathrm{cross}}.
\]
定义两层跨层融合的等变特征
\[
\Phi_{\mathrm{cross}}^{(a,b)}:=\mathrm{TP}^{\oplus}_{a,b}(u,u)\in W_{\mathrm{cross}}.
\]
当任务需要标量/不变量特征时，再选择任意不变量化算子 \(\mathrm{Inv}:W_{\mathrm{cross}}\to\mathbb{R}^m\) 并读出 \(\mathrm{Inv}(\Phi_{\mathrm{cross}}^{(a,b)})\)。在坐标展开下，\(\Phi_{\mathrm{cross}}^{(a,b)}\) 的双线性展开包含同层项与交叉项（来自 \(V_a\) 与 \(V_b\) 的混合分量），从而在不增加深度的前提下显式引入跨层耦合信息。为避免与下文“层对限制算子”记号冲突，本文保留 \(\mathrm{TP}_{a,b}:V_a\times V_b\to W_{\mathrm{cross}}\) 表示从 \(\mathrm{TP}_{\mathrm{cross}}\) 限制得到的层对双线性算子。

（任意 \(L\) 层版本，与实现中的 `num_interaction=L` 对齐）
为避免与 Definition 5' 中的“邻居数 \(n:=|\mathcal{N}(i)|\)”混淆，本文用 \(L\ge 2\) 表示 interaction 层数（网络深度），并与实现参数对应为 \(L:=\texttt{num\_interaction}\)。
得到等变特征序列
\[
h^{(1)},h^{(2)},\dots,h^{(L)},\qquad
h^{(t)}\in V_t:=\bigoplus_{l=0}^{l_{\max}}\left(\mathbb{R}^{c_l^{(t)}}\otimes \mathcal{H}_l\right),\quad t=1,\dots,L.
\]
记直和拼接（concatenation / direct sum embedding）
\[
h^{(\le L)}:=h^{(1)}\oplus\cdots\oplus h^{(L)}\in V_{1:L}:=V_1\oplus\cdots\oplus V_L.
\]

（等变跨层融合特征：抽象表述）
选择一个目标表示空间 \(W_{\mathrm{cross}}\)（可为不可约表示的直和，亦可随层对变化；其类型与维度均为超参数），并选定一个等变双线性算子（Definition 4）：
\[
\mathrm{TP}_{\mathrm{cross}}:\ V_{1:L}\times V_{1:L}\to W_{\mathrm{cross}},
\qquad
\mathrm{TP}_{\mathrm{cross}}(D(R)u,\ D(R)v)=D_{W}(R)\,\mathrm{TP}_{\mathrm{cross}}(u,v).
\]
则可定义跨层融合的等变特征
\[
\Phi_{\mathrm{cross}}^{(L)}:=\mathrm{TP}_{\mathrm{cross}}\!\left(h^{(\le L)},\,h^{(\le L)}\right)\in W_{\mathrm{cross}}.
\]
当任务需要标量输出（如能量）时，可进一步在 \(W_{\mathrm{cross}}\) 上施加任意不变读出/不变量化算子 \(\mathrm{Inv}:W_{\mathrm{cross}}\to \mathbb{R}^m\)，得到不变量特征向量 \(\mathrm{Inv}(\Phi_{\mathrm{cross}}^{(L)})\)。在 irreps 记号下，“投影到平凡表示（\(l=0\)）”是 \(\mathrm{Inv}\) 的一个特例；本文为保持实现无关性不固定 \(\mathrm{Inv}\) 的形式。

给定任意一组层对集合 \(\mathcal{P}\subseteq\{(a,b):1\le a<b\le L\}\)。由 Lemma 8（直和分解），\(\mathrm{TP}_{\mathrm{cross}}\) 可限制到任意层对得到双线性算子
\[
\mathrm{TP}_{a,b}:V_a\times V_b\to W_{\mathrm{cross}},\qquad \mathrm{TP}_{a,b}(u,v):=\mathrm{TP}_{\mathrm{cross}}(\iota_a u,\ \iota_b v),
\]
其中 \(\iota_a:V_a\hookrightarrow V_{1:L}\) 为直和嵌入。定义层对 \((a,b)\) 的等变跨层特征
\[
s_{a,b}:=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})\in W_{\mathrm{cross}}.
\]
当任务需要标量读出时，进一步定义
\[
\psi_{a,b}:=\mathrm{Inv}(s_{a,b})\in \mathbb{R}^m,
\]
并据此定义用于标量读出的“多层跨层特征向量”
\[
\Psi_{\mathcal{P}}^{(L)}:=\Big[\ \{\psi_{t,t}\}_{t=1}^{L},\ \{\psi_{a,b}\}_{(a,b)\in\mathcal{P}}\ \Big].
\]
特别地，为与实现中“跨层特征通道数随 \(L\) 线性增长”的设计对齐，我们采用“以末层为锚点（anchor）”的层对集合
\[
\mathcal{P}_{\mathrm{anchor}}^{(L)}:=\{(t,L):\ 1\le t\le L-1\}.
\]
对应的跨层标量特征为 \(\{\psi_{t,L}\}_{t=1}^{L-1}\)。若同时包含同层自项 \(\{\psi_{t,t}\}_{t=1}^{L}\)，则将二者拼接记为
\[
\Psi_{\mathrm{anchor}}^{(L)}:=\Big[\ \{\psi_{t,t}\}_{t=1}^{L},\ \{\psi_{t,L}\}_{t=1}^{L-1}\ \Big].
\]
存在实现方式在每个 \((t,L)\) 处预留固定宽度的**跨层特征通道**（其宽度记为超参数 \(d_{\mathrm{cross}}\)；通道对应的表示类型同样为超参数，允许为任意不可约表示的直和）。在“每对层输出同一宽度 \(d_{\mathrm{cross}}\)”的设定下，总输出维度为 \((L-1)\,d_{\mathrm{cross}}\)；亦可允许 \(d_{\mathrm{cross}}\) 依赖于层对 \((t,L)\)，此时总维度为 \(\sum_{t=1}^{L-1} d_{\mathrm{cross}}^{(t,L)}\)。

### Theorem 1（体阶上界：跨层双线性等变特征）
若 \(\mathrm{bo}(h^{(a)})\le K_a\)、\(\mathrm{bo}(h^{(b)})\le K_b\)，则跨层双线性等变特征满足
\[
\mathrm{bo}(s_{a,b})=\mathrm{bo}(\mathrm{TP}_{a,b}(h^{(a)},h^{(b)}))\le K_a+K_b.
\]
进一步，对任意不变量化算子 \(\mathrm{Inv}\)（线性），有
\[
\mathrm{bo}(\psi_{a,b})=\mathrm{bo}(\mathrm{Inv}(s_{a,b}))\le K_a+K_b.
\]
因此在固定深度与有限截断配置下，跨层融合可在读出端引入体阶上界为 \(K_a+K_b\) 的跨深度双线性特征；其可达性由 Theorem 2 给出。

**Proof.** 由 Lemma 2 直接得 \(\mathrm{bo}(s_{a,b})\le K_a+K_b\)。又因 \(\mathrm{Inv}\) 为线性映射，不会引入新的索引依赖，故 \(\mathrm{bo}(\mathrm{Inv}(s_{a,b}))\le \mathrm{bo}(s_{a,b})\)。□

### Corollary 1（任意 \(L\) 层：跨层集合的体阶上界）
在 Definition 6 的任意 \(L\) 层设定下，设对每个 \(t=1,\dots,L\) 有 \(\mathrm{bo}(h^{(t)})\le K_t\)。
则对任意层对集合 \(\mathcal{P}\subseteq\{(a,b):a<b\}\)，有
\[
\mathrm{bo}(s_{a,b})\le K_a+K_b,\qquad \forall (a,b)\in\mathcal{P},
\]
并且（若取不变量化读出）\(\mathrm{bo}(\psi_{a,b})\le K_a+K_b\)。从而 \(\Psi_{\mathcal{P}}^{(L)}\) 的每个坐标分量体阶均被对应的 \(K_a+K_b\) 上界控制。
特别地，对 anchor 集合 \(\mathcal{P}_{\mathrm{anchor}}^{(L)}\) 有
\[
\mathrm{bo}(s_{t,L})\le K_t+K_L,\qquad t=1,\dots,L-1,
\]
因此
\[
\max_{t\le L-1}\mathrm{bo}(s_{t,L})\le \max_{t\le L-1}(K_t+K_L).
\]

### Theorem 2（可达性：存在构造使体阶达到 \(K_a+K_b\)）
在以下非退化条件下，存在特征 \(h^{(a)},h^{(b)}\) 使得
\[
\mathrm{bo}(s_{a,b})=\mathrm{bo}(\mathrm{TP}_{a,b}(h^{(a)},h^{(b)}))=K_a+K_b.
\]
**条件（充分条件）**：
1. 存在两个互不相交的邻居子集 \(S,T\subseteq \mathcal{N}(i)\) 满足 \(|S|=K_a, |T|=K_b\)
2. 存在非零标量函数 \(\phi_S\in \mathcal{H}^{(|S|)}\) 与 \(\psi_T\in \mathcal{H}^{(|T|)}\)，分别仅依赖于 \(S\) 与 \(T\)（等价地：对所有真子集 \(U\subsetneq S\) 有 \(\mathbb{E}[\phi_S\mid U]=0\)，对所有真子集 \(U\subsetneq T\) 有 \(\mathbb{E}[\psi_T\mid U]=0\)）
3. 存在某个线性泛函 \(\ell:W_{\mathrm{cross}}\to\mathbb{R}\)，使得标量双线性型
\[
\tilde{\mathcal{I}}(u,v):=\ell(\mathrm{TP}_{a,b}(u,v))
\]
在所考虑的有限维特征空间上非退化（等价地：在某一组坐标下可写为 \(\tilde{\mathcal{I}}(u,v)=u^\top M v\) 且 \(M\) 可逆）。
4. 假设 A2–A3 成立

**Proof.** 上界 \(\mathrm{bo}(s_{a,b})\le K_a+K_b\) 已由 Theorem 1 给出。下面证明下界。

由条件 3，存在可逆矩阵 \(M\) 使得在某一组坐标下
\[
\tilde{\mathcal{I}}(u,v)=\ell(\mathrm{TP}_{a,b}(u,v))=u^\top M v.
\]
令 \(A:=I\)、\(B:=M\)，则 \(\tilde{\mathcal{I}}(u,v)=\langle Au,\ Bv\rangle\)，其中 \(\langle\cdot,\cdot\rangle\) 为标准点积，且 \(A,B\) 可逆。

取任意非零 \(\phi_S\in \mathcal{H}^{(|S|)}\)、\(\psi_T\in \mathcal{H}^{(|T|)}\)（分别仅依赖于 \(S\) 与 \(T\)），并定义向量值函数
\[
\tilde u_S(x):=\phi_S(x_S)e_1,\qquad \tilde v_T(x):=\psi_T(x_T)e_1,
\]
其中 \(e_1\) 为第一标准基向量。令
\[
u_S:=A^{-1}\tilde u_S,\qquad v_T:=B^{-1}\tilde v_T.
\]
则 \(u_S\)（分别 \(v_T\)）仍仅依赖于 \(S\)（分别 \(T\)），且其各坐标分量仍属于 \(\mathcal{H}^{(|S|)}\)（分别 \(\mathcal{H}^{(|T|)}\)）。
由此得到标量函数
\[
f:=\ell(\mathrm{TP}_{a,b}(u_S,v_T))=\tilde{\mathcal{I}}(u_S,v_T)=\phi_S(x_S)\psi_T(x_T).
\]
由 Lemma 5（并使用 A2–A3），\(f\in\mathcal{H}^{(|S|+|T|)}\) 且 \(P_{|S|+|T|}f=f\)。由于 \(\phi_S,\psi_T\) 非零，故 \(\|P_{K_a+K_b}f\|_{L^2}>0\)，从而由 Lemma 3 得 \(\mathrm{bo}(f)\ge K_a+K_b\)。
又因 \(f=\ell(s_{a,b})\) 是 \(s_{a,b}\) 的某个线性坐标（线性泛函作用），由 Definition 5 的向量值扩展有 \(\mathrm{bo}(s_{a,b})\ge \mathrm{bo}(f)\)。因此 \(\mathrm{bo}(s_{a,b})\ge K_a+K_b\)。与上界合并得到 \(\mathrm{bo}(s_{a,b})=K_a+K_b\)。□

### Corollary 2（任意 \(L\) 层：对任意选定层对的可达性）
在 Definition 6 的任意 \(L\) 层设定下，固定任意一对 \((a,b)\)（\(1\le a<b\le L\)）。
若 Theorem 2 的非退化条件对该对 \((a,b)\) 成立（即存在互不相交子集 \(S,T\subseteq\mathcal{N}(i)\) 以及相应纯成分函数等），则存在一组 \(L\) 层特征 \(\{h^{(t)}\}_{t=1}^L\)（其余层可取为零或任意不影响的特征），使得
\[
\mathrm{bo}(s_{a,b})=\mathrm{bo}(\mathrm{TP}_{a,b}(h^{(a)},h^{(b)}))=K_a+K_b.
\]
因此，只要跨层集合 \(\mathcal{P}\) 包含该对 \((a,b)\)，则读出端可获得体阶可达 \(K_a+K_b\) 的跨层双线性等变特征；若进一步施加不变量化 \(\mathrm{Inv}\) 且其不退化条件满足（见 Theorem 2 条件 3 的讨论），则可得到体阶同样可达的标量特征分量。

> 备注（与“每个 anchor 层对输出固定宽度跨层特征”的一致性）：当采用 anchor 集合 \(\mathcal{P}_{\mathrm{anchor}}^{(L)}\) 时，上述结论对任意 \((t,L)\)（\(t<L\)）逐对成立；因此即使只显式包含 \(L-1\) 个跨层对，也能在固定深度下为“早期层 \(\rightarrow\) 末层”的多种体阶组合提供可达性保证。该结论与每对层分配多少个跨层特征维度（\(d_{\mathrm{cross}}\)）无关。

### Corollary 3（固定深度与有限截断配置下的标量特征子空间扩展：任意 \(L\) 层）
设共有 \(L\) 个 interaction（对应实现中的 `num_interaction=L`），并记最终层为 \(h^{(L)}\)。
为讨论标量任务，固定某个不变量化算子 \(\mathrm{Inv}\)。基线模型仅从最终层构造同层二次标量特征 \(\psi:=\psi_{L,L}=\mathrm{Inv}(\mathrm{TP}_{L,L}(h^{(L)},h^{(L)}))\) 并读出 \(E=\rho(\psi)\)。

跨层融合模型在读出端加入多层跨层特征（先等变双线性、再不变量化；Definition 6）：
\[
E'=\rho'\!\left(\psi\ \oplus\ \Psi_{\mathcal{P}}^{(L)}\right),
\]
其中 \(\mathcal{P}\) 可取任意层对集合（例如 anchor 集合 \(\mathcal{P}_{\mathrm{anchor}}^{(L)}\)）。
则由 Lemma 6，函数类满足 \(\mathcal{F}\subseteq \mathcal{F}'\)（扩展模型包含基线模型）。

进一步，由 Theorem 1–2 及 Corollary 1–2，\(\mathcal{F}'\) 可在**不增加消息传递深度 \(L\)** 的前提下，在读出端包含更高体阶（至某个 \(K_a+K_b\)，其中 \((a,b)\in\mathcal{P}\)）的跨层双线性特征；在不变量化读出下，这将扩展可表示的标量特征成分。

### Proposition 2（逐层递推 vs. 跨层加和：体阶上界的对照刻画）
本小节比较两类机制在体阶上界上的结论：其一为逐层消息传递的递推上界；其二为跨层双线性等变特征（及其不变量化读出）的加和式上界。

为统一记号，记
\[
K_t:=\mathrm{bo}\!\left(h^{(t)}\right),\qquad t=1,\dots,L.
\]

**(A) 逐层消息传递：递推上界**  
设存在一族逐邻居项 \(g_j^{(t)}\) 与逐层非线性 \(\sigma_t\)，使得
\[
h^{(t)}=\sigma_t\!\left(\sum_{j\in\mathcal{N}(i)} g_j^{(t)}\right),
\]
并且对每个 \(t\ge 1\)，各 \(g_j^{(t)}\) 的每个坐标分量的体阶上界满足
\[
\mathrm{bo}\big((g_j^{(t)})_q\big)\le K_{t-1}+1,\qquad \forall j,\ \forall q.
\]
则由 Lemma 1（求和闭包）可得
\[
K_t\le K_{t-1}+1,
\]
从而递推得到
\[
K_L\le K_0+L,
\]
其中 \(K_0\) 为“初始特征”的体阶上界（例如仅依赖单个邻居的边特征对应 \(K_0=1\) 的情形）。

> 注：该结论为上界；是否取等号需要额外的非退化条件。

**(B) 跨层双线性等变特征：加和式上界与可达性**  
由 Theorem 2 与 Corollary 2，当读出端显式包含某对层 \((a,b)\) 的跨层双线性等变特征 \(s_{a,b}\) 时，存在构造使得
\[
\mathrm{bo}(s_{a,b})=K_a+K_b,
\qquad a<b\le L,
\]
并且有上界 \(\mathrm{bo}(s_{a,b})\le K_a+K_b\)（Theorem 1）。
因此，对给定层对集合 \(\mathcal{P}\) 而言，读出端可达的“最大体阶”满足
\[
K_{\max}^{\mathrm{cross}}(\mathcal{P})\;\ge\;\max_{(a,b)\in\mathcal{P}}(K_a+K_b),
\]
并且在 Theorem 2 的非退化条件对达到该最大值的层对成立时，上界可达。

特别地，若 \(\mathcal{P}\) 允许取遍所有层对（\(\mathcal{P}=\{(a,b):a<b\}\)）且 \(K_t\) 随 \(t\) 单调不减，则
\[
\max_{a<b}(K_a+K_b)=K_L+K_{L-1}.
\]
若进一步满足 \(K_t=K_0+t\)（即上式对所有 \(t\) 取等号），则
\[
K_L+K_{L-1}=2K_L-1,
\]
从而相较于仅依赖末层读出的 \(K_L\)，加和式组合可给出更高的体阶上界。

---

## 5. 与 MACE 的对照：有限截断配置下的标量读出差异（不变量化后）

### 5.1 有限截断配置下的多体标量特征生成（不变量化后）
在固定截断配置下，MACE 可被抽象为基于等变张量积的可学习 \(N\)-body 展开（与 ACE/MTP 的不变量代数相关）。在该设定下，以下截断参数被固定：
- 角动量截断 \(l_{\max}\)
- 每个 \(l\) 的 multiplicity（通道数）\(\{c_l\}\)
- 允许的张量积路径集合（\(l_1\otimes l_2\to l_3\) 的子集）
- interaction blocks 的数目（深度）

因此，在给定有限截断配置下，MACE 所能生成的不变量集合可定义为某个有限维线性子空间 \(\mathcal{V}_{\text{MACE}}(l_{\max},\{c_l\},\text{paths},L)\subset \mathcal{V}_{\text{inv}}\)。本文仅讨论固定截断配置下的不变量子空间结构差异。

### 5.1.1 MACE 的多体标量特征构造机制（不变量化后）

MACE 在单个 interaction block 内的多体不变量构造遵循以下步骤：

1. **等变特征的算子链表示（intertwiner chain）**：在固定截断配置下，令
   \[
   V^{(t)}:=\bigoplus_{l=0}^{l_{\max}}\left(\mathbb{R}^{c_l^{(t)}}\otimes \mathcal{H}_l\right)
   \]
   表示第 \(t\) 层的等变特征空间。令 \(U\) 表示用于耦合的边/径向特征所张成的（有限维）等变空间（其具体形式由径向基、球谐/方向基以及元素嵌入决定；在固定截断配置下视为给定）。一次层内张量积（耦合）可抽象为一个 \(G=\mathrm{SO}(3)\) 的 intertwiner：
   \[
   T^{(t)}: V^{(t-1)}\otimes U \to V^{(t)}.
   \]
   在不可约表示分量上，\(T^{(t)}\) 由 Clebsch–Gordan 系数给出，并满足选择规则 \(|l_1-l_2|\le l_3\le l_1+l_2\) 与 \(l_3\le l_{\max}\)。若一个 interaction block 内包含多次耦合，则可表示为算子复合
   \[
   h^{(t)}=T^{(t)}_{m}\circ \cdots \circ T^{(t)}_{2}\circ T^{(t)}_{1}(h^{(t-1)}),
   \]
   其中每个 \(T^{(t)}_{k}\) 具有上述 intertwiner 结构。

2. **\(l=0\) 投影与 ACE 型不变量基（coupling graph）**：记 \(P^{(0)}:V^{(t)}\to (\mathbb{R}^{c_0^{(t)}}\otimes \mathcal{H}_0)\cong \mathbb{R}^{c_0^{(t)}}\) 为投影到标量不可约表示（\(l=0\)）的正交投影。则在给定耦合路径（即一组有限的耦合图/耦合树，coupling graph）\(\Gamma\) 与径向/角向截断索引集合 \(\Lambda\) 下，可将层内多次耦合与最终 \(l=0\) 投影写为
   \[
   \psi_{\Gamma,\Lambda}^{(t)}:=P^{(0)}\Big(T^{(t)}_{m}\circ \cdots \circ T^{(t)}_{1}(h^{(t-1)})\Big),
   \]
  其中 \((\Gamma,\Lambda)\) 唯一确定了耦合顺序、不可约表示耦合的中间角动量通道、以及径向基/元素通道的取值。由此得到的 \(\{\psi_{\Gamma,\Lambda}^{(t)}\}\) 构成在固定截断配置下的 ACE 型不变量基（或其有限维张成空间）；模型读出可写为该有限维不变量空间上的可学习映射（线性组合与后续 MLP 等）。

3. **体阶与截断约束**：层内多次 TP 允许生成高体阶项，但在有限截断配置下，实际可生成的不变量集合受到如下约束：
   - 角动量截断 \(l_{\max}\)：限制了可生成的不可约表示类型
   - TP 路径集合：实际实现中仅允许部分 TP 路径组合
   - 通道数 \(\{c_l\}\)：限制了不变量子空间的维度

因此，MACE 在固定截断配置下生成的不变量子空间为
\[
\mathcal{V}_{\text{MACE}} = \text{span}\{\psi_{\text{MACE}}^{(t)} : t=1,\ldots,L\},
\]
其维度与体阶上界受上述截断参数约束。

### 5.1.2 同层多次张量积（MACE/ACE 耦合图）下的体阶上界与可达性
本小节给出一个与 “同层输出进行多次张量积（multiple tensor products within one block）” 直接对应的体阶刻画，用于与第 5.2 节的跨层融合作对照。

为抽象 MACE/ACE 的 “耦合图（coupling graph）” 机制，考虑单个 interaction block 内以某个等变“局部密度/局部基”特征为原子因子（factor）的构造。令 \(\rho\) 表示该因子特征（可向量值/等变），并记其体阶为
\[
K_\rho:=\mathrm{bo}(\rho).
\]
耦合图 \(\Gamma\) 的一个关键组合度参数是其相关阶/耦合次数，记为
\[
\nu(\Gamma)\in\mathbb{N}_{\ge 1},
\]
表示该不变量（或等变输出）在结构上包含 \(\nu(\Gamma)\) 个因子 \(\rho\) 的多线性耦合（在 irreps 语言中对应多次张量积并沿某条耦合树逐步耦合）。

在此抽象下，\(\Gamma\) 诱导一个 \(\nu(\Gamma)\)-线性的（等变或不变）映射
\[
F_\Gamma:\underbrace{V_\rho\times\cdots\times V_\rho}_{\nu(\Gamma)\ \text{次}}\to W_\Gamma,
\]
并产生同层耦合输出 \(F_\Gamma(\rho,\ldots,\rho)\)（若 \(W_\Gamma\) 为平凡表示则为不变量；否则为等变输出；二者均可通过后续读出用于能量/力等任务）。

#### Lemma 7（多线性映射的体阶上界相加）
设 \(F:V_1\times\cdots\times V_m\to W\) 为任意 \(m\)-线性映射（等变/不变与否不影响体阶结论）。若输入特征满足 \(\mathrm{bo}(u_i)\le K_i\)，则
\[
\mathrm{bo}\big(F(u_1,\ldots,u_m)\big)\le \sum_{i=1}^{m}K_i.
\]
**Proof.** 对 \(m=2\) 为 Lemma 2。对任意 \(m\ge 2\)，结论可由归纳证明：将 \(F(\cdot,\ldots,\cdot)\) 视为关于最后一个自变量的线性映射，其系数为前 \(m-1\) 个自变量的多线性表达，并在每一步应用 Lemma 2 与 Definition 5 的并集界即可。□

#### Proposition 5（同层多次张量积：体阶随相关阶线性增长）
在上述耦合图抽象下，若 \(\mathrm{bo}(\rho)\le K_\rho\)，则对任意耦合图 \(\Gamma\) 有
\[
\mathrm{bo}\big(F_\Gamma(\rho,\ldots,\rho)\big)\le \nu(\Gamma)\,K_\rho.
\]
**Proof.** 由 Lemma 7，取 \(m=\nu(\Gamma)\) 且各输入均为 \(\rho\)，得 \(\mathrm{bo}(F_\Gamma(\rho,\ldots,\rho))\le \sum_{i=1}^{\nu(\Gamma)}K_\rho=\nu(\Gamma)K_\rho\)。□

#### Remark（可达性：\(\mathrm{bo}(F_\Gamma(\rho,\ldots,\rho))=\nu(\Gamma)\) 的充分条件）
当 \(\rho\) 为“单邻居因子”（即 \(K_\rho=1\)）且耦合映射 \(F_\Gamma\) 在所考虑的有限维特征子空间上非退化，并且存在 \(\nu(\Gamma)\) 个两两不交的邻居子集作为各因子的支撑集合时，可将 Theorem 2 的构造推广至 \(\nu(\Gamma)\) 因子情形，从而证明 \(\mathrm{bo}(F_\Gamma(\rho,\ldots,\rho))=\nu(\Gamma)\) 的可达性（推广要点同 Lemma 5：互不相交变量上的纯成分乘积保持纯成分且体阶相加）。

#### 对比：同层多次张量积 vs. 跨层融合
- **同层多次张量积（MACE/ACE）**：在固定深度 \(L\) 下，单个 block 的可达体阶受相关阶 \(\nu(\Gamma)\) 控制，并随 \(\nu(\Gamma)\)（在本文设定下）至多线性增长（Proposition 5）。该机制通过“同层内部的多次耦合”提升体阶。
- **跨层融合（本文）**：跨层项的体阶上界与可达性由 Theorem 1–2 给出，为 \(K_a+K_b\) 的加和式组合；其提升来自“把不同深度已获得的体阶成分做双线性组合”，而不是增加同层内部的耦合次数。
- **二者关系**：在固定截断配置下，两者的生成元集合不满足全序包含关系。固定 \(\nu(\Gamma)\) 与 \(L\) 时，跨层融合对应于在不改变同层耦合图集合的条件下引入额外生成元；改变 \(\nu(\Gamma)\) 则对应于改变同层多线性耦合的相关阶，从而改变同层可达体阶上界及其对应的特征子空间。

### 5.2 跨层融合：跨深度双线性等变特征（及其不变量化读出）

跨层融合通过引入跨深度的双线性**等变**特征
\[
s_{a,b}:=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})\in W_{\mathrm{cross}},\qquad a<b,
\]
扩展读出端可用的特征集合，其中 \(h^{(a)},h^{(b)}\) 来自不同深度 \(a < b\)。当任务需要标量输出时，再施加不变量化算子 \(\mathrm{Inv}\) 得到 \(\psi_{a,b}:=\mathrm{Inv}(s_{a,b})\in\mathbb{R}^m\)。

**构造机制**：跨层融合直接对不同深度的等变特征进行双线性等变构造 \(s_{a,b}\)。在不同坐标实现（irreps / Cartesian / ICTD / pure-Cartesian）下，假设 A1 保证对任意线性泛函 \(\ell\)（包括不变量化输出坐标）都有有限项坐标乘积的表示。若进一步对 \(s_{a,b}\) 施加不变量化 \(\mathrm{Inv}\)，则得到用于标量任务的 \(\psi_{a,b}\)。该构造与 MACE 的标量读出同属于“等变耦合后施加不变量化”的范畴；差异仅在于自变量分别来自不同深度 \(a<b\) 的特征空间与同一深度的特征空间。

**体阶性质**：由 Theorem 1，若 \(\mathrm{bo}(h^{(a)})\le K_a\)、\(\mathrm{bo}(h^{(b)})\le K_b\)，则
\[
\mathrm{bo}(s_{a,b})\le K_a+K_b,
\qquad
\mathrm{bo}(\psi_{a,b})\le K_a+K_b.
\]
在 Theorem 2 的非退化条件下，存在构造使得 \(\mathrm{bo}(s_{a,b})=K_a+K_b\)。

**标量特征子空间扩展（在固定 \(\mathrm{Inv}\) 下）**：在相同 \(L,l_{\max},\{c_l\}\) 与路径截断配置下，跨层融合构造的标量特征子空间为
\[
\mathcal{V}_{\text{extended}} = \mathcal{V}_{\text{MACE}} + \text{span}\{\psi_{a,b} : 1\le a < b \le L\},
\]
其中 \(+\) 表示子空间的并（span）。若存在 \((a,b)\) 使得 \(\psi_{\text{cross}}^{(a,b)} \notin \mathcal{V}_{\text{MACE}}\)，则 \(\dim(\mathcal{V}_{\text{extended}}) > \dim(\mathcal{V}_{\text{MACE}})\)。

### 5.2.1 与 MACE 的严格对照（生成元集合与函数类包含）

为避免将差异归因于“是否能够构造多体”，以下在固定截断配置下以生成元集合、特征空间与最优逼近误差为对象给出可验证的结论。

设第 \(t\) 层等变特征在固定截断配置下取值于有限维表示空间 \(V_t\)。为将结论与标量任务对齐，本小节固定某个不变量化算子 \(\mathrm{Inv}\) 并仅讨论其输出张成的标量特征空间。记同层二次标量生成集合
\[
\mathcal{G}_{\mathrm{self}}(t):=\{\mathrm{Inv}(\mathrm{TP}_{t,t}(u,u)):\ u\in V_t\},
\]
以及跨层二次不变量生成集合
\[
\mathcal{G}_{\mathrm{cross}}(a,b):=\{\mathrm{Inv}(\mathrm{TP}_{a,b}(u,v)):\ u\in V_a,\, v\in V_b\},\qquad a<b.
\]

在给定的数据分布（或经验测度）诱导的内积 \(\langle \cdot,\cdot\rangle_{L^2}\) 下，设基线模型在读出端所使用的不变量特征空间为某个有限维线性子空间
\[
\mathcal{W}_{\mathrm{base}}\subset \mathrm{span}(\mathcal{G}_{\mathrm{self}}(L)).
\]
跨层融合在读出端加入跨层生成元后，对应的不变量特征空间定义为
\[
\mathcal{W}_{\mathrm{cross}}:=\mathcal{W}_{\mathrm{base}}+\mathrm{span}(\mathcal{G}_{\mathrm{cross}}(a,b)),
\]
其中 \(+\) 表示线性张成意义下的子空间和。

**Proposition 3（函数类包含）**：若读出网络类在特征输入维度扩展后保持不变且具有充分表达性，则由 \(\mathcal{W}_{\mathrm{base}}\) 扩展至 \(\mathcal{W}_{\mathrm{cross}}\) 诱导函数类包含关系。

**Proof.** 取读出网络参数使其忽略新增特征即可（Lemma 6）。□

**Proposition 4（最优 \(L^2\) 逼近误差的严格下降：充分条件）**：设目标函数 \(f\in L^2\)，并令
\[
e_{\mathrm{base}}:=\inf_{g\in \mathcal{W}_{\mathrm{base}}}\|f-g\|_{L^2},\qquad
e_{\mathrm{cross}}:=\inf_{g\in \mathcal{W}_{\mathrm{cross}}}\|f-g\|_{L^2}.
\]
则 \(e_{\mathrm{cross}}\le e_{\mathrm{base}}\)。进一步，若存在 \(w\in \mathrm{span}(\mathcal{G}_{\mathrm{cross}}(a,b))\) 使得其在 \(\mathcal{W}_{\mathrm{base}}\) 的正交补上的分量与 \(f\) 在该正交补上的分量具有非零相关（例如 \(\langle P_{\mathcal{W}_{\mathrm{base}}^\perp}f,\ P_{\mathcal{W}_{\mathrm{base}}^\perp}w\rangle_{L^2}\neq 0\)），则 \(e_{\mathrm{cross}}< e_{\mathrm{base}}\)。

**Proof.** 由 \(\mathcal{W}_{\mathrm{base}}\subseteq \mathcal{W}_{\mathrm{cross}}\) 立得 \(e_{\mathrm{cross}}\le e_{\mathrm{base}}\)。令 \(r:=P_{\mathcal{W}_{\mathrm{base}}^\perp}f\) 为基线残差。若存在 \(w\) 使得 \(P_{\mathcal{W}_{\mathrm{base}}^\perp}w\) 与 \(r\) 非正交，则沿该方向对 \(P_{\mathcal{W}_{\mathrm{base}}}f\) 作一次线性修正可严格降低残差范数，故最优误差严格下降。□

**判据（经验秩判据：可检验的严格扩展充分条件）**：在有限样本 \(\{x_n\}_{n=1}^N\) 上，令 \(\Phi_{\mathrm{base}}\in\mathbb{R}^{N\times d}\) 为基线不变量特征矩阵，\(\Phi_{\mathrm{cross}}\in\mathbb{R}^{N\times d'}\) 为加入跨层特征后的特征矩阵。若
\[
\mathrm{rank}(\Phi_{\mathrm{cross}})>\mathrm{rank}(\Phi_{\mathrm{base}}),
\]
则 \(\mathcal{W}_{\mathrm{cross}}\) 在经验内积意义下严格扩展 \(\mathcal{W}_{\mathrm{base}}\)；并且若目标 \(f\) 在样本上的残差向量与新增特征张成的子空间存在非零相关，则经验最小二乘误差严格下降。

### 5.3 关于优化性质的补充说明
跨层融合在结构上引入对中间层特征的额外读出路径，从而改变梯度传播路径。下面给出一个与具体实现无关、仅依赖于“读出端包含双线性跨层项”的链式法则刻画，从而得到浅层特征获得额外梯度分量的充分条件。

#### 5.3.1 梯度路径的链式法则刻画（任意双线性跨层项）
为简化记号，令
\[
u:=h^{(\le L)}\in V_{1:L}=V_1\oplus\cdots\oplus V_L,
\]
并考虑一个读出端包含的跨层双线性项
\[
s:=\mathcal{B}(u,u)\in W,
\]
其中 \(\mathcal{B}:V_{1:L}\times V_{1:L}\to W\) 为双线性映射（可等变或不变；见 Definition 6 与 Lemma 8），\(W\) 为某个输出表示空间。
令 \(\Gamma:W\to \mathcal{Z}\) 表示后续任意可微的“读出/变换”（例如用于能量任务时，可包含不变量化与 MLP 读出；本文不固定其形式），并令损失为
\[
\mathcal{L}:=\ell\!\left(\Gamma(s),\,y\right),
\]
其中 \(y\) 为监督信号，\(\ell\) 为可微损失函数。

固定 \(u\)，对任意扰动 \(\delta u\in V_{1:L}\)，由双线性可得
\[
\delta s
=D(\mathcal{B}(u,u))[\delta u]
=\mathcal{B}(\delta u,u)+\mathcal{B}(u,\delta u).
\]
取 \(V_{1:L}\) 与 \(W\) 上任意内积并用 \((\cdot)^\ast\) 表示相应的伴随算子。令
\[
g:=\left(D\Gamma(s)\right)^\ast\left(\nabla_{\Gamma(s)}\ell\right)\in W
\]
为跨层项在 \(W\) 上的“反传信号”（即把损失梯度通过 \(\Gamma\) 反传到 \(s\) 的结果）。则
\[
\delta\mathcal{L}
=\langle g,\ \delta s\rangle_W
=\langle g,\ \mathcal{B}(\delta u,u)+\mathcal{B}(u,\delta u)\rangle_W.
\]
注意到对固定 \(u\)，映射 \(\delta u\mapsto \mathcal{B}(\delta u,u)\) 与 \(\delta u\mapsto \mathcal{B}(u,\delta u)\) 均为线性算子 \(V_{1:L}\to W\)。因此存在（且由内积唯一确定）其伴随算子 \((\mathcal{B}(\cdot,u))^\ast:W\to V_{1:L}\)、\((\mathcal{B}(u,\cdot))^\ast:W\to V_{1:L}\)，使得
\[
\delta\mathcal{L}
=\left\langle (\mathcal{B}(\cdot,u))^\ast g + (\mathcal{B}(u,\cdot))^\ast g,\ \delta u \right\rangle_{V_{1:L}}.
\]
从而得到跨层双线性项对 \(u\) 的梯度表达式
\[
\nabla_{u}\mathcal{L}\Big|_{\text{via }s}
=(\mathcal{B}(\cdot,u))^\ast g + (\mathcal{B}(u,\cdot))^\ast g.
\]
进一步，记 \(P_a:V_{1:L}\to V_a\) 为直和上的正交投影，则对任意层 \(a\) 有
\[
\nabla_{h^{(a)}}\mathcal{L}\Big|_{\text{via }s}
=P_a\Big((\mathcal{B}(\cdot,u))^\ast g + (\mathcal{B}(u,\cdot))^\ast g\Big).
\]
该式表明：只要读出端的损失显式依赖于跨层双线性项 \(s=\mathcal{B}(u,u)\)，则每一层特征 \(h^{(a)}\) 都会获得一个由 \(g\) 与当前特征 \(u\) 共同决定的梯度分量；该分量并不需要“先通过末层再经由层间递推”的单一路径。

从实践角度看，跨层融合提供的直接梯度路径不仅可能改善优化条件，更重要的是为不同深度特征提供差异化的学习信号：浅层特征更易获得与局部几何相关的梯度，而深层特征更易获得与全局模式相关的梯度，从而促使模型学习更丰富的分层表征。

#### 5.3.2 与“仅末层读出”的对比（梯度链的乘积结构）
作为对照，若基线结构的读出仅依赖最终层 \(h^{(L)}\)（记 \(\mathcal{L}=\ell(\tilde\Gamma(h^{(L)}),y)\)），则对早期层 \(a<L\)，梯度必经由从第 \(a\) 层到第 \(L\) 层的复合雅可比/伴随的链式传递：
\[
\nabla_{h^{(a)}}\mathcal{L}
=\left(D h^{(L)} / D h^{(a)}\right)^\ast \nabla_{h^{(L)}}\mathcal{L},
\]
其中 \(D h^{(L)} / D h^{(a)}\) 是网络在层间复合得到的 Fréchet 导数（在参数固定时可写为一串雅可比的复合）。若存在 \(\alpha\in(0,1)\) 使得 \(\left\|\left(D h^{(L)} / D h^{(a)}\right)^\ast\right\|\le \alpha^{L-a}\)，则早期层梯度满足指数衰减上界；若存在 \(\beta>1\) 使得 \(\left\|\left(D h^{(L)} / D h^{(a)}\right)^\ast\right\|\ge \beta^{L-a}\)，则梯度下界呈指数放大。二者均会改变梯度幅值的尺度，从而影响优化条件数。

跨层融合的差异在于：在上述“仅末层读出”的梯度项之外，额外引入了 \(\nabla_{h^{(a)}}\mathcal{L}\big|_{\text{via }s}\) 这一直接项。若 \(\mathcal{B}\) 通过 Lemma 8 的分解被结构性地约束为仅依赖某些层对（例如 anchor 层对 \((t,L)\)），则该直接项进一步分解为这些层对分量贡献的和。

---

## 6. 与实现的一致性说明（Implementation Mapping）

本文的抽象算子覆盖下列实现（不依赖于具体的 irreps 记号；允许输出为任意等变表示，并可在需要时再做不变量化）：
- **e3nn（irreps）**：`FullyConnectedTensorProduct` / `ElementwiseTensorProduct`（等变双线性；分量耦合由 CG 系数给出）+ 可选的不变量化读出
- **partial Cartesian（strict/loose）**：在 Cartesian 坐标组织张量，同时借助 e3nn 的 irreps 分解与 Clebsch–Gordan 系数实现（可能带截断/近似的）耦合与投影；该实现仅覆盖部分通道/路径，应视为 **partial Cartesian** + 可选的不变量化读出
- **PureCartesian（dense/sparse）**：按 rank 的 δ/ε contraction 与 Gram 型 contraction（在 Cartesian 坐标系下实现双线性耦合；可等变/不变）+ 可选的不变量化读出
- **ICTD（harmonic polynomial / harmonic tensor）**：以调和多项式/调和张量为基在 Cartesian 坐标系下实现张量积（等价于 irreps 张量积的一种坐标实现）；并可通过收缩或线性不变量化算子得到标量读出 + 可选的不变量化读出

### Lemma 8（直和空间上的双线性型分解：层对分量）
设 \(V_{1:L}=V_1\oplus\cdots\oplus V_L\)。令 \(\mathcal{B}:V_{1:L}\times V_{1:L}\to W\) 为任意双线性映射，其中 \(W\) 为某个（可选的）输出表示空间；当关心等变性时，可令 \(W\) 携带表示 \(D_W(R)\)。
则存在唯一的一族双线性映射 \(\{\mathcal{B}_{a,b}:V_a\times V_b\to W\}_{1\le a,b\le L}\)（定义为对直和嵌入的限制 \(\mathcal{B}_{a,b}(u,v):=\mathcal{B}(\iota_a u,\ \iota_b v)\)），使得对任意
\(u=\oplus_{t}u_t\in V_{1:L}\)、\(v=\oplus_{t}v_t\in V_{1:L}\) 成立
\[
\mathcal{B}(u,v)=\sum_{a=1}^{L}\sum_{b=1}^{L}\mathcal{B}_{a,b}(u_a,v_b).
\]
若 \(\mathcal{B}\) 进一步满足等变性 \(\mathcal{B}(D(R)u,D(R)v)=D_W(R)\mathcal{B}(u,v)\)，则每个 \(\mathcal{B}_{a,b}\) 亦等变；若 \(\mathcal{B}\) 满足不变性（\(D_W\) 为平凡表示的特例），则每个 \(\mathcal{B}_{a,b}\) 亦不变。

**Proof.** 直和分解的双线性展开（对每个自变量分别按直和分解展开）给出上式；唯一性来自于 \(\mathcal{B}_{a,b}\) 的定义即为对嵌入子空间的限制。若 \(\mathcal{B}\) 等变，则对任意 \(R\in G\)，有
\(\mathcal{B}_{a,b}(D_a(R)u,\ D_b(R)v)=\mathcal{B}(\iota_a D_a(R)u,\ \iota_b D_b(R)v)=D_W(R)\mathcal{B}(\iota_a u,\ \iota_b v)=D_W(R)\mathcal{B}_{a,b}(u,v)\)；不变情形为 \(D_W(R)=I\) 的特例。□

**与实现的对应**：
当实现中设置 `num_interaction=L` 时，存在实现方式先做特征拼接
\(f_{\mathrm{combine}}=\mathrm{cat}(f_1,\ldots,f_L)\)（对应 \(h^{(\le L)}\in V_{1:L}\) 的坐标拼接表示），再通过某个双线性模块输出一个跨层融合特征
\[
s=\mathcal{B}\!\left(h^{(\le L)},\,h^{(\le L)}\right)\in W,
\]
其中输出表示空间 \(W\)（以及其坐标维度）是实现层面的超参数：可以是任意不可约表示的直和；也可以在后续再施加 \(\mathrm{Inv}:W\to\mathbb{R}^m\) 得到用于标量任务的不变量特征。

由 Lemma 8，\(s\) 总可以分解为各层对分量 \(\mathcal{B}_{a,b}(h^{(a)},h^{(b)})\) 的（\(W\)-值）和：
\[
s=\sum_{a,b}\mathcal{B}_{a,b}\!\left(h^{(a)},h^{(b)}\right).
\]
因此，单个“在直和上作用的”双线性模块在数学上对应于**对所有层对分量的某种线性组合/汇聚**；若希望严格实现 Definition 6 中某个特定层对集合（例如 anchor 集合 \(\mathcal{P}_{\mathrm{anchor}}^{(L)}\)）的“显式拼接”，则需在结构上对 \(\mathcal{B}\) 施加块稀疏约束（例如令 \(\mathcal{B}_{a,b}\equiv 0\) 对所有不在目标集合中的 \((a,b)\)），或直接按层对分别计算并拼接对应的跨层特征通道。

### 6.1 两阶段跨层融合-再融合（与 FusedSCEquiTensorPot 的实现结构对齐）
考虑 FusedSCEquiTensorPot 的一类实现：在读出前包含两次双线性张量积/耦合。其数学抽象为：

1. **第一阶段（跨层融合）**：将所有层特征拼接为 \(u:=h^{(\le L)}\in V_{1:L}\)，并计算
\[
s^{(1)} := \mathcal{B}_1(u,u)\in W_1,
\]
其中 \(\mathcal{B}_1:V_{1:L}\times V_{1:L}\to W_1\) 为双线性映射，\(W_1\) 为实现层面的超参数输出表示空间（可等变或不变，亦可在后续做不变量化）。

2. **第二阶段（再融合）**：把第一阶段输出与原拼接特征再拼接
\[
v := u \oplus s^{(1)} \in V_{1:L}\oplus W_1,
\]
并再次施加双线性映射
\[
s^{(2)} := \mathcal{B}_2(v,v)\in W_2.
\]
最后对 \(s^{(2)}\) 施加后续读出（例如不变量化算子与 MLP 复合）得到标量输出。

#### Proposition 6（两阶段双线性读出的体阶上界：抽象形式）
设 \(K_{\max}:=\max_{t\le L}\mathrm{bo}(h^{(t)})\)。则
\[
\mathrm{bo}(s^{(1)})\le 2K_{\max},
\qquad
\mathrm{bo}(s^{(2)})\le 4K_{\max}.
\]
**Proof.** 由 Lemma 2，双线性给出 \(\mathrm{bo}(s^{(1)})=\mathrm{bo}(\mathcal{B}_1(u,u))\le \mathrm{bo}(u)+\mathrm{bo}(u)\le 2K_{\max}\)。
又因 \(v=u\oplus s^{(1)}\) 的每个坐标分量体阶上界为 \(\max(\mathrm{bo}(u),\mathrm{bo}(s^{(1)}))\le 2K_{\max}\)，再次应用 Lemma 2 得
\(\mathrm{bo}(s^{(2)})=\mathrm{bo}(\mathcal{B}_2(v,v))\le 2\cdot (2K_{\max})=4K_{\max}\)。□

> 备注：该上界刻画了在不增加消息传递深度 \(L\) 的前提下，读出端通过两次双线性耦合可提高体阶上界；与之对照，MACE/ACE 的同层多次张量积提升体阶主要由相关阶 \(\nu(\Gamma)\) 控制（见 Proposition 5）。

#### Example 1（参数设定：与 MACE 的体阶上界对比）
下面给出一个参数设定下的体阶上界对比。该对比仅使用本文已给出的体阶上界引理，因此给出严格上界；达到上界需要额外的非退化条件。

**示例设定说明**：本例中 MACE 采用 \(\nu=3\) 的耦合图仅为示意性设定。实际 MACE 实现可能采用不同的相关阶与路径截断策略，体阶上界相应变化。本比较旨在展示两种机制在相同深度下**体阶增长模式**的结构差异：MACE 的体阶上界主要由层内多线性耦合的相关阶控制（线性于 \(\nu\) 的增长），而跨层融合通过跨深度双线性项实现加和式组合（形如 \(K_a+K_b\)）。

**设定**：
- **MACE**：interaction blocks 数 \(L=2\)，同层耦合图的相关阶（correlation order）取 \(\nu=3\)。
- **FusedSCEquiTensorPot（本文模型）**：interaction 数 \(L=2\)，读出端包含**两阶段双线性耦合**（见第 6.1 节）：先对拼接特征 \(u=h^{(\le L)}\) 施加一次双线性映射得到 \(s^{(3)}\)，再将 \(v=u\oplus s^{(3)}\) 再次输入双线性映射得到 \(s^{(5)}\)。
- 取初始因子体阶上界 \(K_0=1\)，并采用 Proposition 2(A) 的递推上界：\(K_t\le K_{t-1}+1\)。

**(i) MACE：同层相关阶 \(\nu=3\) 的上界传播**
在第一个 interaction block 内，按 Proposition 5（取 \(K_\rho=K_0=1\)）有
\[
\mathrm{bo}(\text{block}_1\ \text{输出})\le \nu K_0 = 3.
\]
若第二个 block 的多线性耦合以“上一 block 的输出特征”作为因子（体阶上界记为 \(K_{\mathrm{in},2}\)），则再次由 Proposition 5（或 Lemma 7 的直接应用）有
\[
\mathrm{bo}(\text{block}_2\ \text{输出})\le \nu K_{\mathrm{in},2}\le 3\times 3=9.
\]
因此，在该设定下，MACE 输出体阶上界不超过 \(9\)。该上界对应于第二个 block 的相关阶多线性耦合以第一个 block 输出作为因子的设定。若第二个 block 的多线性耦合因子并非第一个 block 的输出，或额外施加路径截断，则需要在该设定下重新给出 \(K_{\mathrm{in},2}\) 的上界并据此更新最终上界。

**(ii) 本文模型：\(L=2\) 且两次双线性读出**
由“逐层至多 +1”的上界，\(K_1\le 2\)、\(K_2\le 3\)，从而 \(K_{\max}\le 3\)。
代入 Proposition 6（两阶段双线性上界）得到
\[
\mathrm{bo}(s^{(1)})\le 2K_{\max}\le 6,
\qquad
\mathrm{bo}(s^{(2)})\le 4K_{\max}\le 12.
\]
因此，在该设定下，两阶段双线性读出的体阶上界不超过 \(12\)。

**对比结论（体阶上界）**：
- MACE（\(L=2,\nu=3\)）在上述设定下的体阶上界为 \(9\)；
- FusedSCEquiTensorPot（\(L=2\)，两阶段双线性读出）在 Proposition 6 的条件下的体阶上界为 \(12\)。

> 备注：以上比较仅涉及可证明的体阶上界，不构成对训练后“达到该上界”的断言。达到上界需要满足相应的非退化条件，且可达性仍受有限 \(l_{\max}\)、有限通道数、路径集合与参数化方式等约束。

---

## 7. 结论性表述

> **命题**：在固定消息传递深度与有限截断配置（例如 \(l_{\max}\)、通道数与张量积路径集合受限）下，跨层融合通过引入跨深度双线性等变特征 \(s_{a,b}=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})\) 扩展读出端可用的特征集合；当任务需要标量输出时，对 \(s_{a,b}\) 施加不变量化算子 \(\mathrm{Inv}\) 得到 \(\psi_{a,b}\) 并输入标量读出网络。该跨层双线性特征的体阶满足上界 \(K_a+K_b\)（Theorem 1），并在非退化条件下可达（Theorem 2）。上述结论刻画了跨层融合在“等变耦合 + 可选不变量化”机制层面的结构差异。

---

## 8. Related Work：现有 MLIP 的跨深度机制分类

本节按数学结构将现有 MLIP 中的“跨层”机制划分为两类，并澄清本文方法的区别所在。

### 8.1 A 类：逐层读出累加（layer-wise additive readout）
该类方法在每个 interaction block（或 message passing layer）后施加读出，并将各层标量贡献累加：
\[
E_i=\sum_{t=1}^{L}\rho_t\!\left(\mathrm{Inv}(h_i^{(t)})\right),
\]
其中 \(\mathrm{Inv}(\cdot)\) 表示任意旋转不变的读出（例如对 \(l=0\) 分量的线性读出，或对等变特征的同层二次不变量），\(\rho_t\) 为标量网络。本文仅将该类结构作为抽象类型陈述，而不对具体文献实现逐一列举。

### 8.2 B 类：跨深度双线性等变特征（及其不变量化读出，本文）
本文方法的核心不同在于显式构造跨深度的双线性等变特征（必要时再做不变量化）：
\[
E_i=\rho\!\left(\ldots,\ \mathrm{Inv}(\mathrm{TP}_{a,b}(h_i^{(a)},h_i^{(b)})),\ \ldots\right),\qquad a<b,
\]
并将其与同层自项共同输入读出网络。由 Lemma 2 及 Theorem 1–2，可得该结构在固定深度与有限截断配置下引入额外的跨层双线性特征；在标量任务下，经不变量化后的特征体阶上界为 \(K_a+K_b\)，并在非退化条件下可达。本文据此将其与逐层读出累加机制区分。

概述：A 类对应逐层读出累加；B 类对应跨深度双线性等变特征（及其不变量化读出）。两者在读出端特征构造方式上存在结构差异。

### 8.3 代表性 MLIP 对比（结构分类，非穷尽）
为比较不同方法的结构差异，本节给出若干代表性 MLIP 的结构分类。下表不穷尽各模型的全部实现变体；同一模型在不同代码库中可能存在“仅末层读出”与“逐层读出累加”等实现差异。本文仅对读出范式作结构性归类，不将该表作为对具体代码库实现的断言。

<table>
<thead>
<tr>
<th>模型/家族</th>
<th>主要对称性/表示</th>
<th>多体构造的主要来源</th>
<th>逐层读出累加（A 类）</th>
<th>跨深度双线性（B 类）</th>
</tr>
</thead>
<tbody>
<tr>
<td>SchNet</td>
<td>SE(3) 平移不变、旋转不变（连续滤波卷积）</td>
<td>多层消息传递的非线性叠加</td>
<td rowspan="7">实现依赖</td>
      <td rowspan="7">在实现中未显式引入</td>
</tr>
<tr>
<td>PhysNet</td>
<td>旋转不变（显式物理先验 + 消息传递）</td>
<td>逐层残差修正与消息传递</td>
</tr>
<tr>
<td>DimeNet / DimeNet++</td>
<td>旋转不变（角向基/三体消息）</td>
<td>多个交互块 + 输出块（output blocks）</td>
</tr>
<tr>
<td>GemNet（系列）</td>
<td>旋转不变（高阶几何消息）</td>
<td>多模块/多输出块</td>
</tr>
<tr>
<td>PaiNN</td>
<td>SO(3) 等变（标量+向量通道）</td>
<td>等变消息传递与非线性</td>
</tr>
<tr>
<td>NequIP / Allegro</td>
<td>E(3) 等变（irreps）</td>
<td>层内 TP + 多层消息传递</td>
</tr>
<tr>
<td>MACE</td>
<td>E(3) 等变（ACE/TP 型多体基）</td>
<td>同层内进行多次张量积（TP）构造多体不变量；通过层内 TP 路径生成 ACE 风格的多体基</td>
</tr>
<tr>
<td>ACE/MTP（显式基）</td>
<td>旋转不变（显式多体不变量基）</td>
<td>显式 N-body 基函数截断</td>
<td>不适用</td>
<td>不适用</td>
</tr>
</tbody>
</table>

**对照要点（与 MACE 的差异表述）**：
1. MACE 的核心优势在于：在单个 interaction block 内通过张量积路径生成 ACE 风格的多体不变量（并在有限 \(l_{\max}\)、有限 multiplicity 与有限路径截断下形成有限维子空间）。
2. 跨层融合在不增加消息传递深度的前提下引入跨深度双线性等变特征 \(s_{a,b}=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})\)，并可在标量任务下通过不变量化算子得到 \(\psi_{a,b}\)，从而在固定截断配置下扩展可生成的标量特征子空间（参见 Lemma 2 / Theorem 2）。

### 8.4 对比结论

> 若干神经网络势能模型采用逐层读出并累加能量贡献（layer-wise additive readout）。该机制与本文所讨论的跨深度双线性机制不同：本文在等变表征 \(h^{(a)},h^{(b)}\) 之间引入 \(s_{a,b}=\mathrm{TP}_{a,b}(h^{(a)},h^{(b)})\)，并可在标量任务下进一步施加不变量化得到 \(\psi_{a,b}\)，从而在固定深度与有限截断配置下扩展标量特征生成集合，并在非退化条件下使体阶上界 \(K_a+K_b\) 可达（Theorem 2）。
