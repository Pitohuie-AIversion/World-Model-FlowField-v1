# The Well `shear_flow` 实测数据审计报告

> 审计文件：`/root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5`  
> 审计日期：2026-09-18  
> 运行环境：NVIDIA vGPU-32GB (CUDA 13.0, PyTorch 2.10.0+cu128)

---

## 1. 根属性与元数据

| 属性名 | 实际读取值 | 说明 |
|---|---|---|
| `dataset_name` | `shear_flow` | 数据集名称 |
| `n_trajectories` | **4** | 该验证集文件中包含的独立初始条件轨迹数 |
| `n_spatial_dims` | 2 | 二维空间 |
| `grid_type` | `cartesian` | 均匀笛卡尔网格 |
| `simulation_parameters` | `['Reynolds', 'Schmidt']` | 动力学控制参数 |
| `Reynolds` | 10000.0 | 雷诺数 |
| `Schmidt` | 0.1 | 施密特数 |

---

## 2. 存储结构与字段键名

实际 HDF5 采用 The Well 统一规范的层级群组（Rank-0 标量场存于 `t0_fields`，Rank-1 向量场存于 `t1_fields`）：

```text
/
├── boundary_conditions/
│   ├── x_periodic/mask    (256,)   [bool]
│   └── y_periodic/mask    (512,)   [bool]
├── dimensions/
│   ├── time               (200,)   [float64] (t in [0.0, 19.9], dt=0.1)
│   ├── x                  (256,)   [float32] (range: [0.0, 1.0])
│   └── y                  (512,)   [float32] (range: [0.0, 1.0])
├── scalars/
│   ├── Reynolds           ()       [float32] = 10000.0
│   └── Schmidt            ()       [float32] = 0.1
├── t0_fields/
│   ├── pressure           (4, 200, 256, 512) [float32]
│   └── tracer             (4, 200, 256, 512) [float32]
└── t1_fields/
    └── velocity           (4, 200, 256, 512, 2) [float32]
```

---

## 2.1 Closure-R4 空间轴与物理尺度契约

真实 HDF5 字段的空间维度按 `(Nx, Ny) = (256, 512)` 存储，且
`ShearFlowDataset` 保持该原生空间顺序，不做转置。因此正式张量契约为：

- `dim -2 = x`
- `dim -1 = y`
- Tensor layout: `(..., C, Nx, Ny)`

HDF5 中 `dimensions/x` 与 `dimensions/y` 的标签值均归一化到
`[0, 1]`。物理空间导数则使用 shear-flow 仿真的实际尺度
`Lx = 1.0, Ly = 2.0`。Closure-R4 的 FFT 梯度、散度、涡量、
Laplacian 与能谱波数全部遵守统一的 `(Lx, Ly)` 约定。

pre-R4 物理算子错误地把 `dim -2` 当作 y、`dim -1` 当作 x。
因此，任何使用非零 divergence/vorticity loss 训练得到的 pre-R4
checkpoint 均不得用于正式物理结论；field-only 权重可以保留，但必须使用
Closure-R4 算子重新计算物理指标。

---

## 3. 物理字段数值分布统计

### 3.1 初始状态 ($t = 0$)
- **速度场分量 0（主流向）**：$\min = -0.3483, \max = 0.4241, \text{mean} = 0.0157, \text{std} = 0.2676$
- **速度场分量 1（横向流）**：$\min = -0.0998, \max = 0.0998, \text{mean} \approx -1.3 \times 10^{-6}, \text{std} = 0.0488$
- **压力场 $p$**：全场严格为 0（吻合 Dedalus 初始压力规范 $\int p = 0$ 且初始设为 0）
- **示踪标量 $s$**：$\min = -0.3483, \max = 0.4241$，与主流向剪切速度场初始完全一致（吻合官方文档“示踪剂初始化匹配剪切速度”）。

### 3.2 演化状态 ($t = 10$)
- **压力场 $p$**：已演化出完整的压力梯度场，范围 $[-0.00735, 0.00694]$，空间均值保持为 0。
- **示踪标量 $s$**：范围 $[-0.3458, 0.4129]$，呈现输运扩散与涡旋卷吸现象。

---

## 4. 显存实测与可行性验证（NVIDIA vGPU-32GB）

在 256 × 512 全分辨率输入下，实测前向传播显存占用：

| 模型模块 | 输入尺寸 | 输出尺寸 | 显存峰值 | 状态 |
|---|---|---|---|---|
| **Autoencoder** (Encoder + Decoder) | `(2, 4, 4, 256, 512)` | `(2, 4, 4, 256, 512)` | **2.86 GB** | 远低于 32GB 限额，极其安全 |
| **Latent ST Transformer** (6层, 8头, 256维) | `(2, 4, 64, 32, 64)` | `(2, 1, 64, 32, 64)` | **9.14 GB** | 占 28.5% 显存，留出 >22GB 用于梯度与反向传播 |

结论：**256 × 512 全分辨率在当前硬件上无需降采样即可直接训练！**

---

## 5. 物理一致性实测：Schmidt 数不变性审计

### 5.1 物理原理
在二维不可压缩 Navier-Stokes 方程中，被动示踪标量 $s$ 满足平流扩散方程 $\partial_t s + \mathbf{u} \cdot \nabla s = D \nabla^2 s$，其中扩散系数 $D = 1 / (Re \cdot Sc)$。示踪标量不对流体施加任何浮力或反作用力。
因此，**在相同雷诺数 $Re$ 和完全相同的初始流场条件（IC）下，不同 Schmidt 数（如 $Sc=0.1$ vs $Sc=1.0$）的样本：其速度场 $(u, v)$ 和压力场 $p$ 必须在全时间序列上严格一致（达数值精度极限），而示踪标量 $s$ 必须因扩散差异展现显著不同。**

### 5.2 匹配初始条件实测对比（File A: Train Sc=0.1 vs File B: Valid Sc=1.0）
实测发现跨文件匹配对：Train Traj 0 与 Valid Traj 2 的初始流场 $t=0$ 完全重合（$|v_A - v_B|_{t=0} < 10^{-15}$）。对比 200 步全时序演化：

| 物理场 | 最大绝对偏差 (Max Diff) | 平均绝对偏差 (Mean Diff) | 审计结论 |
|---|---|---|---|
| **水平速度 $u$** | $1.192 \times 10^{-7}$ | $5.204 \times 10^{-12}$ | **严格一致 (PASS)** |
| **竖直速度 $v$** | $5.960 \times 10^{-8}$ | $1.374 \times 10^{-12}$ | **严格一致 (PASS)** |
| **压力 $p$** | $2.980 \times 10^{-8}$ | $1.646 \times 10^{-12}$ | **严格一致 (PASS)** |
| **示踪标量 $s$** | **0.5386** | **0.0660** | **呈现显著物理扩散差异 (PASS)** |

**结论**：流场数据严格遵守不可压缩流体被动示踪标量输运物理守恒律。

---

## 6. 初始条件分布与划分数据泄漏审计

在对 HDF5 文件跨子集分析中发现：
1. **官方划分（Official Split）跨 Sc 存在动力学泄漏**：
   - `valid/shear_flow_Reynolds_1e4_Schmidt_1e0.hdf5` 中的 Traj 0、Traj 1、Traj 2，分别与 `train/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5` 中的 Traj 26、Traj 29、Traj 0 具有完全相同的初始条件流场与全生命周期速度/压力演化。
   - 若直接使用 The Well 官方划分进行跨 Sc 的多任务训练与验证，验证集的速度压力动力学已被训练集“完全记忆”。
2. **架构规范对策**：
   - 必须采用 [DEVELOPMENT_V1.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/DEVELOPMENT_V1.md) 规划的 **分组划分（Grouped Split）**，将相同初始条件的全部 Sc 轨迹强制分配到同一集合中，杜绝动力学泄漏。

---

## 7. Stage 1 审计审查结论

- **任务状态**：**PASS**
- **已确立项**：
  1. 状态定义 $q_t = [u, v, p, s]$（4通道，前两通道为速度分量，后两通道为标量场）
  2. 条件定义 $c = [Re, Sc]$（标量参数，对数尺度适配）
  3. 网格尺寸 $256 \times 512$（周期笛卡尔网格，$\Delta t=0.1, T=200$）
  4. 物理被动标量一致性验证完成，各项数据结构定义已全部对齐。

