# 流场世界模型 V1 开发说明

> 数据集：The Well / `shear_flow`  
> 任务：二维不可压缩流场与被动示踪标量的长期未来状态预测  
> 当前状态：V1 开发规范（审查修订版）  
> 
> 📚 **核心文档导航**：
> - [正式论文实验章节与三 Seed 出版级结果](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/MANUSCRIPT_RESULTS.md)
> - [Closure-R4 空间轴序契约与物理资产治理](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/CLOSURE_R4_SPATIAL_AXIS_FIX.md)
> - [系统架构说明与技术规范](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/ARCHITECTURE.md)
> - [项目工程 TodoList 与研发进度看板](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/TODOLIST.md)
> - [基准评测报告与物理指标分析](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/BENCHMARK_RESULTS.md)
> - [V1 阶段全链验收报告与 10 月任务规划](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/V1_ACCEPTANCE_REPORT.md)
> - [The Well 数据集与物理协议说明](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/DATASET.md)
> - [数据实测审计报告](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/DATA_AUDIT.md)


## 1. 目标

V1 先把世界模型的动力学核心做稳定。输入是一段历史物理状态和物理参数，输出是未来一段时间的状态序列。

状态定义为

\[
q_t=[u_t,v_t,p_t,s_t]
\]

其中 `u, v` 是二维速度分量，`p` 是压力，`s` 是被动示踪标量。

> **字段映射说明**：The Well 数据集中速度场存储为 `velocity`（向量场，含 `u_x` 和 `u_z` 两个分量），本文统一用 `u, v` 分别指代水平和竖直速度分量。生成脚本中竖直轴标记为 `z`，官方文档页面标记为 `y`。实际 HDF5 字段键名和轴序以数据审计结果为准，此处的 `u, v` 仅为本文符号约定。

物理条件定义为

\[
c=[Re,Sc]
\]

其中 `Re` 是雷诺数，`Sc` 是施密特数。

预测任务写成

\[
\hat q_{t+1:t+H}
=
F_\theta(q_{t-L+1:t},Re,Sc)
\]

`L` 表示历史窗口长度，`H` 表示自由滚动长度。

V1 默认设定：

- **历史窗口**：$L = 4$
- **单步训练目标**：$H = 1$
- **短程自由滚动训练**：$H \in \{2, 4, 8\}$（课程式递增）
- **独立评价滚动长度**：$h \in \{1, 5, 10, 20, 30\}$

V1 不处理复杂几何、机器人动作、自然语言、随机潜变量、任意坐标解码和三维流场。这些能力等到当前动力学链路跑通后再接入。

## 2. 数据集

### 2.1 数据源

V1 使用 The Well 的 `shear_flow`。

根据 The Well 官方文档，该数据集的已知规格如下：

| 属性 | 值 |
|---|---|
| 空间分辨率 | **256 × 512**（$N_y \times N_x$） |
| 时间步数 | **200** |
| 时间步长 | $\Delta t = 0.1$（仿真时间单位） |
| 时间范围 | $t_{\min} = 0$，$t_{\max} = 20$ |
| 空间域 | $0 \le x \le 1$（水平），$-1 \le y \le 1$（竖直） |
| 网格类型 | 均匀笛卡尔 |
| 边界条件 | **周期** |
| 可用字段 | tracer（标量）、pressure（标量）、velocity（向量） |
| 压力约束 | $\int p = 0$（零积分压力规范） |
| Reynolds 取值 | `[1e4, 5e4, 1e5, 5e5]` → **4 个** |
| Schmidt 取值 | `[0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]` → **7 个** |
| 初始条件参数 | $n_{\text{shear}} \in [2,4]$，$n_{\text{blobs}} \in [2,3,4,5]$，$w \in [0.25, 0.5, 1.0, 2.0, 4.0]$ → $2 \times 4 \times 5 = $ **40 个** |
| **PDE 参数组合** | $4 \times 7 = 28$ 个 |
| **总轨迹数** | $28 \times 40 = $ **1120 条** |
| **总数据量** | **~547 GB** |
| 求解器 | Dedalus（谱方法），双精度 |

物理方程：

- 不可压缩 Navier-Stokes：粘性系数 $\nu = 1 / Re$
- 被动示踪标量输运：扩散系数 $D = \nu / Sc = 1 / (Re \cdot Sc)$

以上数据来自 The Well 官方文档，仅作参考。正式训练前必须读取实际下载文件，记录真实的字段键名、轴序、张量形状和划分方式，并与上述规格做一致性核对。

### 2.2 数据存储策略

当前数据盘 `/root/autodl-tmp` 总容量 350 GB，全量数据集 547 GB 无法完整落盘。V1 采取以下策略：

**方案：选取参数子集**

选取 2 个 Reynolds × 4 个 Schmidt × 40 个初始条件 = **320 条轨迹**（约 ~156 GB），具体参数组合：

- $Re \in \{1\text{e}4, 1\text{e}5\}$（一低一高，覆盖层流和湍流过渡区间）
- $Sc \in \{0.1, 1.0, 5.0, 10.0\}$（覆盖低扩散到高扩散全范围）

参数留出划分留出的 Re / Sc 从上述子集选取，不依赖未下载的数据。

如果实际下载后磁盘紧张，进一步缩减至 2 Re × 3 Sc × 40 IC = 240 条轨迹。

### 2.3 数据一致性检查

被动示踪标量不反作用于速度和压力动力学。相同 `Re` 和相同初始流场条件下，只改变 `Sc` 时，`u/v/p` 轨迹可能相同或高度一致，而 `s` 会发生变化。

第一项数据检查就是验证这一点。检查完成前，不开始正式模型比较。

### 2.4 三套数据划分

项目保留三套划分，分别回答不同问题。

1. **官方划分**  
   用于和 The Well 公开基线比较。

2. **分组划分**  
   相同 `Re` 和相同初始条件的样本放入同一集合，避免不同 `Sc` 下的相关速度压力轨迹同时进入训练集和测试集。

3. **参数留出划分**  
   整个 `Re` 或 `Sc` 取值从训练集留出，用于测试参数泛化。

`Re` 泛化和 `Sc` 泛化分别报告。`Re` 会影响速度压力动力学，`Sc` 主要改变示踪标量扩散，两者不能合成一个总泛化指标。

## 3. 六阶段架构

V1 保持六阶段结构。

```text
阶段 1  物理输入
   ↓
阶段 2  空间潜状态表示
   ↓
阶段 3  物理条件适配
   ↓
阶段 4  潜空间动力学
   ↓
阶段 5  物理解码
   ↓
阶段 6  训练监督与独立评价
```

阶段 1 至阶段 5 是推理主链。阶段 6 负责训练损失和独立评价。

## 4. 阶段 1：物理输入

模型输入为

\[
q_{t-L+1:t}
\]

其中

\[
q_t=[u,v,p,s]
\]

条件输入为

\[
c=[Re,Sc]
\]

同时保留 `x, y, Δt` 和周期边界信息。

本轮不启用以下分支：

- 几何 / SDF
- 点云
- 机器人动作
- 稀疏传感器
- 自然语言提示
- 符号偏微分方程提示
- 外部视觉观测

原因很简单：`shear_flow` 没有这些变量的变化或配对监督。

## 5. 阶段 2：空间潜状态表示

V1 使用确定性空间编码器

\[
Z_t=E_\phi(q_t)
\]

并保留空间结构

\[
Z_t\in\mathbb{R}^{H_z\times W_z\times C_z}
\]

不能把整个流场压成单个全局向量。

### 5.1 编码器架构

编码器采用基于卷积的 UNet 编码器路径（不含跳跃连接的下采样路径），将 256×512×4 的物理场逐级下采样至潜空间：

| 参数 | 默认值 | 说明 |
|---|---|---|
| 输入分辨率 | 256 × 512 | $N_y \times N_x$ |
| 输入通道 | 4 | $[u, v, p, s]$ |
| 空间下采样倍率 | ×8 | 3 层 stride-2 卷积 |
| 潜状态分辨率 | **32 × 64** | $H_z \times W_z$ |
| 潜通道数 | **64** | $C_z$，后续作为超参数调整 |
| 潜状态 token 数 | 2048 | $32 \times 64 = 2048$，用于 Transformer |

如果 32×64 显存仍然紧张，降级方案为 ×16 下采样（16×32 = 512 token）。

### 5.2 解码器架构

解码器为编码器的对称上采样路径，使用转置卷积或上采样 + 卷积，将 32×64×64 恢复至 256×512×4。

解码器满足

\[
\tilde q_t=D_\psi(Z_t)
\]

第一轮先检查表示质量，再检查预测能力。重建误差低只能说明编码器保留了当前场信息，不能说明该潜状态适合长期预测。

需要比较两种训练方式：

- **冻结表示**：先训练编码器和解码器，随后冻结它们，只训练动力学模型。
- **联合微调**：在未来状态预测和自由滚动训练中继续更新编码器和解码器。

如果联合微调明显改善长期预测，说明预测需要的潜状态和单纯重建需要的潜状态并不完全相同。

V1 不使用 VAE、向量量化、扩散潜变量和几何图编码器。

Sparse2Full 仓库中已有多种空间模型实现（UNet、FNO、SwinIR 等），编码器/解码器实现可复用其骨干代码，不从零编写。

## 6. 阶段 3：物理条件适配

`Re` 和 `Sc` 先经过一个小型 MLP

\[
e_c=\mathrm{MLP}(Re,Sc)
\]

再通过 AdaLN（自适应层归一化）注入 Transformer。

`Re` 和 `Sc` 在输入 MLP 前做对数变换：$[\log Re, \log Sc]$，避免量级差异过大（$Re$ 范围 $10^4$ 至 $5 \times 10^5$，$Sc$ 范围 $0.1$ 至 $10$）。

第一版只使用 AdaLN。FiLM、条件 token 和条件交叉注意力不同时叠加。

为了判断显式物理条件是否真的有帮助，所有主要模型都保留两套输入协议：

### 协议 A：只输入状态

\[
q_{history}\rightarrow q_{future}
\]

### 协议 B：状态加物理条件

\[
(q_{history},Re,Sc)\rightarrow q_{future}
\]

对比两套协议后，再判断条件编码是否带来增益。

## 7. 阶段 4：潜空间动力学

### 7.1 主模型

主模型是带条件的因子化时空潜空间 Transformer。

输入

\[
Z_{t-L+1:t}
\]

条件

\[
[Re,Sc]
\]

空间注意力和时间注意力分开处理：

```text
历史潜状态
    ↓
空间注意力
    ↓
时间注意力
    ↓
空间注意力
    ↓
未来潜状态
```

这样可以控制时空 token 的计算量，同时保留空间和时间交互。

### 7.2 下一状态预测形式

V1 不预设残差预测一定优于直接预测。两种形式都保留。

直接预测：

\[
\hat Z_{t+1}
=
T_\theta(Z_{t-L+1:t};Re,Sc)
\]

残差预测：

\[
\Delta\hat Z_{t\rightarrow t+1}
=
T_\theta(Z_{t-L+1:t};Re,Sc)
\]

\[
\hat Z_{t+1}
=
Z_t+\Delta\hat Z_{t\rightarrow t+1}
\]

两种方式使用相同数据和训练预算比较。

### 7.3 潜空间自由滚动

推理时，预测结果直接写回历史潜状态缓冲区：

```text
Z(t-L+1:t)
    ↓
Transformer
    ↓
Ẑ(t+1)
    ↓
更新历史潜状态缓冲区
    ↓
Transformer
    ↓
Ẑ(t+2)
    ↓
...
```

自由滚动过程中不执行 `潜状态 → 解码 → 再编码 → 潜状态`。

## 8. 阶段 5：物理解码

未来潜状态解码为

\[
\hat q_{t+h}
=
D_\psi(\hat Z_{t+h})
\]

其中

\[
\hat q=[\hat u,\hat v,\hat p,\hat s]
\]

### 8.1 压力处理

先记录原始预测压力的空间均值

\[
\mu_p=\langle\hat p\rangle
\]

随后做零均值投影

\[
\hat p
\leftarrow
\hat p-\langle\hat p\rangle
\]

投影后的压力用于场误差和物理评价。投影前的均值单独保存，用于检查压力漂移。

压力零均值投影只在 **解码后的评价阶段** 执行，不放入训练计算图（避免在自由滚动中每步投影引入不连续性）。Direct Transformer 直接在物理场上滚动时，同样只在最终评价时执行投影。

### 8.2 压力状态消融

压力虽然放在当前状态中，但它和速度、示踪标量的动力学角色不同。后续保留一个消融：

- `q=[u,v,p,s]`
- 主动力学状态使用 `[u,v,s]`，压力作为辅助输出

这项实验不阻塞第一版实现。

V1 不使用 INR、任意坐标查询、超分辨率解码和力积分。

## 9. 阶段 6：训练目标

训练顺序先解决长期动力学，再加入物理约束。

### 9.1 场值损失与优化空间契约

\[
L_{\mathrm{field}}
=
L_u+L_v+L_p+L_s
\]

每项使用 **均方误差（MSE）**。

> **优化空间与通道权重契约（Protocol P1-1）**：
> - **默认空间（Normalized Feature Space）**：在不可压缩剪切流中，各物理场方差悬殊（例如 $\mathrm{Var}(u) \sim 0.5$ 而 $\mathrm{Var}(p) \sim 10^{-4}$）。在反归一化物理空间直接求和会导致压力与示踪剂梯度被速度场淹没。因此，`train_forecaster.py` 默认在标准化特征空间优化（相当于以通道方差倒数 $1/\sigma_c^2$ 加权的马氏物理距离），亦可通过 `--field_loss_space physical` 切换为未加权物理空间。
> - **物理微分损失空间（Physical Space）**：与场值损失不同，速度散度损失 $L_{\mathrm{div}}$ 与涡量损失 $L_\omega$ 具备严格的流体力学守恒量纲，**必须且始终在反归一化后的真实物理空间计算**。
> - **压力零均值规范（Pressure Gauge Policy, P1-2）**：Decoder 在潜空间输出原始未约束场（`project_pressure=False`）。在反归一化至真实物理空间后，统一施加零空间均值投影 $\int p \, \mathrm{d}\Omega = 0$，消除压力标度不定性。

数据归一化方式：**per-channel per-dataset mean/std 标准化**。在训练集上计算每个通道的全局均值和标准差，写入配置文件，验证集和测试集使用相同统计量。归一化和反归一化必须通过单元测试验证一致性。

通道权重：V1 第一轮使用等权重。如果压力的量级与速度差异过大导致训练不稳定，在第二轮引入通道权重 $[w_u, w_v, w_p, w_s]$ 作为超参数。

### 9.2 短程自由滚动损失

只做单步训练容易在多步预测中积累误差，因此 V1 必须加入短程自由滚动训练。

\[
L_{\mathrm{roll}}
=
\sum_{h=1}^{K}
w_h
L_{\mathrm{field}}
(\hat q_{t+h},q_{t+h})
\]

权重策略：**均匀权重** $w_h = 1/K$。如果后续发现远步精度需要强调，改为线性递增 $w_h = h / \sum_{h'} h'$。

训练长度按以下顺序增加：

```text
1 步
↓
2 步
↓
4 步
↓
8 步
```

每个阶段训练至收敛后再切换到下一阶段。30 步用于独立评价，不作为第一轮长展开训练目标。

### 9.3 散度损失

\[
L_{\mathrm{div}}
=
\left\|
\frac{\partial\hat u}{\partial x}
+
\frac{\partial\hat v}{\partial y}
\right\|_2^2
\]

该损失只作用于速度场。空间导数使用周期 FFT 计算。

### 9.4 涡量一致性损失

二维涡量为

\[
\omega
=
\frac{\partial v}{\partial x}
-
\frac{\partial u}{\partial y}
\]

损失写成

\[
L_\omega
=
\|\hat\omega-\omega_{\mathrm{GT}}\|_2^2
\]

涡量从速度场实时计算（使用 FFT 导数），不依赖数据集中是否预存涡量字段。$\omega_{\text{GT}}$ 从真值速度场计算。

不使用

\[
\|\hat\omega\|^2
\]

作为涡量保持损失，因为该形式会直接压低涡量幅值。

### 9.5 第一轮训练消融

| 实验 | 损失 |
|---|---|
| E0 | `L_field` |
| E1 | `L_field + λ_roll L_roll` |
| E2 | `E1 + λ_div L_div` |
| E3 | `E1 + λ_ω L_ω` |
| E4 | `E1 + λ_div L_div + λ_ω L_ω` |

初始超参数：$\lambda_{\text{roll}} = 1.0$，$\lambda_{\text{div}} = 0.1$，$\lambda_\omega = 0.1$。如果物理损失量级与场值损失相差超过 10 倍，按比例调整系数。

先判断 rollout-aware training 是否改善长期预测，再判断物理损失是否进一步改善对应物理指标。

### 9.6 后置损失

第一版暂不加入：

- 完整 Navier-Stokes 残差
- 被动示踪输运方程残差
- 频谱损失
- 固壁无滑移损失

周期边界数据不使用固壁损失。

## 10. 独立评价

所有模型使用同一套评价代码。

### 10.1 场值指标

分别报告 `u, v, p, s`：

- RMSE
- 相对 L2 误差

### 10.2 速度场物理指标

报告：

\[
\|\nabla\cdot\hat{\mathbf u}\|
\]

以及

\[
\|\hat\omega-\omega_{\mathrm{GT}}\|
\]

### 10.3 全局统计量

动能：

\[
K(t)
=
\frac12\langle u^2+v^2\rangle
\]

涡量平方积分：

\[
\Omega(t)
=
\frac12\langle\omega^2\rangle
\]

能谱：

\[
E(k,t)
\]

周期规则网格优先使用 FFT 导数和频谱分析。

### 10.4 示踪标量指标

报告：

- 示踪标量场误差
- 空间均值
- 方差
- 均值守恒误差
- 方差衰减过程

### 10.5 自由滚动长度

至少报告

\[
h=1,5,10,20,30
\]

重点看误差和物理量如何随预测长度变化。

### 10.6 计算代价

为了判断潜空间是否有实际价值，还要记录：

- 参数量
- 单步推理时间
- 30 步自由滚动时间
- 峰值显存
- 潜状态 token 数
- 训练吞吐率

## 11. 对照模型

基线和主模型尽量使用相同数据版本、历史窗口、归一化方式、预测长度和评价代码。

### B0：持久性预测

\[
\hat q_{t+h}=q_t
\]

### B1：FNO

FNO（傅里叶神经算子）作为神经算子基线。可复用 Sparse2Full 中已有的 FNO2D 实现。

### B2：Direct ST Transformer

直接在物理场 patch 上学习时空预测：

\[
q_{history}
\rightarrow
\mathrm{PatchEmbed}
\rightarrow
\mathrm{ST\ Transformer}
\rightarrow
q_{future}
\]

### P0：Latent ST Transformer

主模型：

\[
q
\rightarrow
E
\rightarrow
Z
\rightarrow
\mathrm{ST\ Transformer}
\rightarrow
D
\rightarrow
q
\]

B2 和 P0 尽量保持相同历史长度、条件输入、注意力结构、参数规模和训练预算，用于单独判断显式潜空间是否有价值。

### V1.1 扩展基线（不阻塞 V1 验收）

以下基线在 V1 代码完成后视时间情况接入：

- **CNextU-Net**：The Well 卷积模型基线，需要确认可用开源实现
- **PDE-Transformer**：需要确认预训练权重可用性，从零训练与微调分别报告

## 12. 关键消融

V1 至少完成以下消融。

### 12.1 是否需要显式潜空间

- Direct ST Transformer (B2)
- Latent ST Transformer (P0)

### 12.2 编码器是否联合微调

- 冻结 Encoder / Decoder
- Encoder / Dynamics / Decoder 联合微调

### 12.3 下一潜状态预测方式

- 直接预测 `Z(t+1)`
- 残差预测 `ΔZ`

### 12.4 条件信息是否有效

- State-only
- State + `Re,Sc`

### 12.5 物理损失是否有效

- E0 至 E4

这些实验优先于大范围超参数搜索。

## 13. 训练流程

### 阶段 A：数据审计

完成：

1. 读取实际 HDF5 元数据，核对与第 2.1 节规格表的一致性
2. 记录实际字段键名、轴序和张量形状
3. 确认时间间隔
4. 确认 `Re` 和 `Sc` 取值
5. 检查不同 `Sc` 下相关 `u/v/p` 轨迹
6. 构建官方划分、分组划分和参数留出划分

### 阶段 B：表示学习

训练

\[
q_t\rightarrow Z_t\rightarrow\tilde q_t
\]

检查 `u, v, p, s` 和涡量重建。

### 阶段 C：单步动力学

训练

\[
Z_{t-L+1:t}
\rightarrow
\hat Z_{t+1}
\]

先确认单步预测有效。

### 阶段 D：短程自由滚动训练

依次训练 2、4、8 步自由滚动。

### 阶段 E：长期自由滚动评价

评价 1、5、10、20、30 步。

### 阶段 F：物理损失消融

按 E0 至 E4 运行。

## 14. 与现有仓库的关系

### 14.1 与 World-Model-FlowField 契约系统

World-Model-FlowField 仓库已实现完整的 Pydantic 契约系统（`WorldState`、`TransitionSample`、`ConditionSeries` 等）。

**V1 决策**：V1 先专注模型实验，数据层使用轻量 PyTorch Dataset / DataLoader，不依赖契约系统。契约集成推迟到 V2，避免同时调试两套系统。

### 14.2 与 Sparse2Full 的代码复用

Sparse2Full 仓库中已有的可复用资产：

| 模块 | Sparse2Full 路径 | V1 用途 |
|---|---|---|
| FNO2D | `models/spatial/fno2d.py` | 基线 B1 |
| UNet | `models/spatial/unet.py` | 编码器/解码器骨干 |
| AR Wrapper | `models/ar/wrapper.py` | 自回归推演参考 |
| 损失函数 | `ops/loss.py`, `ops/losses.py` | 场值损失参考 |
| 性能分析 | `utils/performance.py` | 计算代价统计 |

V1 从 Sparse2Full 迁移所需模块，不引入整个仓库作为依赖。

## 15. 推荐仓库结构

```text
flow-world-model/
├── README.md
├── configs/
│   ├── data/
│   ├── model/
│   ├── train/
│   └── experiment/
├── src/
│   ├── data/
│   │   ├── shear_flow_dataset.py
│   │   ├── normalization.py
│   │   ├── splits.py
│   │   └── windows.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   ├── conditioning.py
│   │   ├── latent_transformer.py
│   │   ├── direct_transformer.py
│   │   └── history_buffer.py
│   ├── losses/
│   │   ├── field.py
│   │   ├── rollout.py
│   │   ├── divergence.py
│   │   └── vorticity.py
│   ├── metrics/
│   │   ├── field.py
│   │   ├── spectral.py
│   │   ├── tracer.py
│   │   ├── compute.py
│   │   └── rollout.py
│   ├── baselines/
│   │   ├── persistence.py
│   │   └── fno.py
│   └── utils/
│       ├── fft_derivatives.py
│       ├── checkpoint.py
│       └── reproducibility.py
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_splits.py
│   ├── train_representation.py
│   ├── train_forecaster.py
│   ├── evaluate_rollout.py
│   └── run_ablation.py
├── tests/
│   ├── test_dataset.py
│   ├── test_splits.py
│   ├── test_encoder_decoder.py
│   ├── test_conditioning.py
│   ├── test_history_buffer.py
│   ├── test_fft_derivatives.py
│   ├── test_losses.py
│   └── test_rollout.py
└── outputs/
    ├── checkpoints/
    ├── metrics/
    └── figures/
```

文件名后续可以调整，但数据、模型、损失、指标、基线和测试的职责保持分开。

## 16. 计算资源与训练预算

### 16.1 硬件环境

| 资源 | 规格 |
|---|---|
| GPU | NVIDIA vGPU-32GB (CUDA 13.0) |
| 数据盘 | `/root/autodl-tmp`，350 GB |
| 系统盘 | 约 30 GB 可用 |

### 16.2 单次训练预算上限

| 参数 | 默认值 |
|---|---|
| batch size | 4（256×512 全分辨率下）|
| 优化器 | AdamW |
| 学习率 | 1e-4（cosine 衰减，warmup 500 步）|
| 训练步数 | 50,000 步（阶段 B 表示学习），100,000 步（阶段 C/D 动力学）|
| 梯度累积 | 2（等效 batch size 8）|
| 混合精度 | bf16 |
| 单次训练最长时间 | **8 小时** |

如果 batch size 4 显存不足，降至 2 并增加梯度累积。

### 16.3 如果 256×512 显存溢出

降级方案：将输入空间分辨率降采样至 128×256（双线性插值），所有模型使用相同降采样版本，评价时上采样回原始分辨率计算指标。此降级在规范中预留，但不是默认行为。

## 17. 最低测试要求

### 数据

必须验证：

- 字段顺序正确
- 时间窗口不跨轨迹
- `Re` 和 `Sc` 与源轨迹对应
- 官方划分、分组划分和参数留出划分可复现
- 分组划分不会把相关 `Sc` 样本拆到训练集和测试集
- 归一化和反归一化一致

### 表示

必须验证：

- 编码器输出尺寸符合配置（32×64×64）
- 解码器恢复原始物理网格（256×512×4）
- 压力零均值投影正确
- 冻结和联合微调两种模式都能运行

### 动力学

必须验证：

- 历史缓冲区删除最旧状态并加入最新预测
- 自由滚动不读取未来真值
- 自由滚动不出现意外的 decode-encode 回环
- Direct 和 Residual 两种潜状态预测头都能运行
- State-only 和 Condition-aware 两种协议输入一致

### 物理算子

必须验证：

- 已知无散场的散度接近 0
- 已知解析场的涡量计算正确
- 周期 FFT 导数与解析周期函数一致

### 评价

必须验证：

- 单步和多步指标对应正确目标帧
- 每个物理量单独报告
- 自由滚动长度与实际自回归步数一致
- 各模型使用相同评价代码
- 计算时间和显存统计方式一致

## 18. V1 验收标准

V1 通过验收时应满足：

1. 已记录实际 `shear_flow` 文件结构和字段，并与第 2.1 节规格表核对一致。
2. 三套数据划分均可复现。
3. 编码器和解码器可以重建 `u, v, p, s`。
4. Transformer 不读取未来真值即可完成单步预测。
5. 已完成 2、4、8 步短程自由滚动训练。
6. 已完成 1、5、10、20、30 步独立评价。
7. Persistence、FNO 和 Direct ST Transformer 通过统一接口评价。
8. 已完成 Direct Transformer 与 Latent Transformer 的公平比较。
9. 已完成 frozen/joint、direct/residual、state-only/condition-aware 消融。
10. 已完成 E0 至 E4 损失消融。
11. 已输出场误差、涡量、散度、动能、涡量平方积分、能谱、示踪标量指标和计算成本。
12. 实验配置、随机种子、模型权重、数据划分和评价结果能够对应保存。

V1 不以主模型超过全部基线为验收条件。主模型如果没有优势，按原实验协议保留结果，并据此修改下一版设计。

## 19. 开发计划

### 19.1 第一阶段：代码与管线就绪（9 月 18 日 → 9 月 30 日）

| 日期 | 任务 | 交付 |
|---|---|---|
| 9 月 18-19 日 | 下载数据子集，审计实际数据文件，检查字段、尺寸、参数和不同 `Sc` 下的相关轨迹 | 数据审计记录 |
| 9 月 20-21 日 | 固定三套划分，完成归一化和窗口生成，通过数据测试 | 可复现数据管线 |
| 9 月 22-23 日 | 实现 Encoder、Decoder、压力投影和表示测试 | 重建基线 |
| 9 月 24 日 | 实现 Re/Sc 条件编码（AdaLN + log 变换） | 条件注入模块 |
| 9 月 25-27 日 | 中秋节假期，不安排人工开发节点 | 已提交计算任务可继续运行 |
| 9 月 28-29 日 | 实现 Latent ST Transformer 和 Direct ST Transformer，单步训练跑通 | 两类 Transformer 可训练 |
| 9 月 30 日 | 实现 Persistence 和 FNO 基线，实现历史缓冲区和自由滚动训练框架 | 代码管线就绪 |

### 19.2 第二阶段：训练与消融（10 月 1 日 → 10 月 12 日）

| 日期 | 任务 | 交付 |
|---|---|---|
| 10 月 1-3 日 | 阶段 B 表示学习训练 + 阶段 C 单步动力学训练 | 单步预测基线 |
| 10 月 4-5 日 | 阶段 D 短程自由滚动训练（2/4/8 步） | rollout-aware 模型 |
| 10 月 6-7 日 | 实现 FFT 导数、物理指标和完整评价管线 | 1/5/10/20/30 步评价 |
| 10 月 8-10 日 | 运行 E0-E4 损失消融、Direct/Latent、Frozen/Joint、Direct/Residual 消融 | 消融训练运行 |
| 10 月 11-12 日 | 汇总全部基线、主模型、消融结果、失败案例和计算成本 | **V1 验收记录** |

### 19.3 消融优先级

如果时间不足以完成全部消融，优先级按以下顺序执行：

1. Direct ST Transformer vs Latent ST Transformer
2. E0 vs E1，先确认 rollout-aware training
3. E1 vs E2/E3/E4，检查物理损失
4. Frozen vs Joint
5. Direct latent prediction vs Residual latent prediction

## 20. 暂缓内容

V1 暂不处理：

- 变化几何
- SDF / 点云编码
- 机器人动作条件
- 传感器同化
- 自然语言查询
- 符号 PDE 条件
- VAE 和其他随机潜变量
- 扩散式动力学
- INR 解码
- 任意分辨率重建
- 完整 Navier-Stokes 残差训练
- 被动示踪输运方程残差训练
- 频谱损失训练
- 三维流场
- 规划和机器人控制
- World-Model-FlowField 契约系统集成
- CNextU-Net 和 PDE-Transformer 基线（移至 V1.1）

这些模块等到相应数据和评价任务明确后再加入。

## 21. V1 要回答的问题

### Q1：显式潜状态有没有必要？

Direct ST Transformer 和 Latent ST Transformer 在尽量相同的条件下比较。如果 latent 不能改善计算成本、长期自由滚动或物理一致性，就不把它作为后续架构前提。

### Q2：自由滚动训练有没有必要？

比较单步训练和 rollout-aware training。重点看误差增长速度，而不是只看一步预测。

### Q3：显式物理条件有没有帮助？

比较 State-only 与 State + `Re,Sc`。同时把 `Re` 泛化和 `Sc` 泛化分开报告。

### Q4：物理损失有没有改善对应物理量？

`L_div` 应改善散度，`L_ω` 应改善涡量结构。如果总损失下降但对应物理指标没有改善，就不能把该损失视为有效。

### Q5：主模型相对现有模型有什么实际收益？

与 FNO 和 Direct ST Transformer 比较：

- 单步精度
- 长期自由滚动
- 物理一致性
- 推理时间
- 显存
- 参数量

最终结论由这些结果决定，不预设潜空间 Transformer 一定更好。
