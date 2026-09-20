# The Well `shear_flow` 数据集与物理协议说明

> **任务定位**：二维不可压缩流场与被动示踪标量的长期未来状态预测  
> **数据源**：The Well 基准数据集（`shear_flow`）  
> **物理仿真器**：Dedalus 谱方法求解器  
> **当前版本**：V1 规范（生产与训练基线）

---

## 1. 物理背景与控制方程

`shear_flow` 数据集模拟的是在双向周期笛卡尔坐标系下受剪切驱动的二维不可压缩流体运动，并同时输运一个无浮力反作用的被动示踪标量（Passive Tracer）。

### 1.1 速度场与压力场（不可压缩 Navier-Stokes 方程）

流体动力学由二维不可压缩 Navier-Stokes 方程严格控制：

$$
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u}
$$

$$
\nabla \cdot \mathbf{u} = 0 \quad (\text{质量守恒 / 无散度约束})
$$

其中：
- $\mathbf{u} = [u, v]^T$ 为流体速度矢量（$u$ 为水平主流向速度，$v$ 为竖直横向速度）；
- $p$ 为运动学压力（Kinematic Pressure）；
- $Re$ 为雷诺数（Reynolds Number），表征惯性力与粘性力的相对强弱。

### 1.2 被动示踪标量场（对流扩散方程）

示踪标量 $s$ 满足对流-扩散方程（Advection-Diffusion Equation）：

$$
\frac{\partial s}{\partial t} + \mathbf{u} \cdot \nabla s = D \nabla^2 s = \frac{1}{Re \cdot Sc} \nabla^2 s
$$

其中：
- $s$ 为被动示踪标量（浓度/染料分布），它不改变流体密度，对流场施加零反作用力；
- $Sc$ 为施密特数（Schmidt Number），定义为动量扩散系数（运动粘度 $\nu = 1/Re$）与质量扩散系数 $D$ 的比值 $Sc = \nu / D$；
- 质量扩散系数为 $D = \frac{1}{Re \cdot Sc}$。

---

## 2. 状态变量与物理条件定义

### 2.1 系统物理状态 $q_t$

在每个离散时间步 $t$，系统完整物理状态包含 4 个物理场通道：

$$
q_t = [u_t, v_t, p_t, s_t] \in \mathbb{R}^{4 \times H \times W}
$$

1. **水平速度 $u$**：主流向剪切速度，初始包含上下反向剪切层；
2. **竖直速度 $v$**：横向扰动速度，触发 Kelvin-Helmholtz 不稳定性与剪切卷吸；
3. **压力场 $p$**：由压力泊松方程解出的椭圆型标量场，具有全场常数不定性（需满足零均值规范）；
4. **示踪标量 $s$**：初始分布与主流向速度对齐，随流场展开复杂拓扑卷吸与分子扩散。

### 2.2 动力学控制参数 $c$

每条仿真轨迹由一对常数标量参数控制：

$$
c = [Re, Sc]
$$

- **雷诺数 $Re$**：典型取值范围涵盖 $10^2 \sim 10^5$；
- **施密特数 $Sc$**：典型取值涵盖 $10^{-1} \sim 10^1$。

#### 参数对数归一化协议
由于 $Re$ 和 $Sc$ 跨越多个数量级，直接线性输入会导致神经网络权重梯度失衡。代码中采用对数归一化映射至标准数值区间：

$$
\tilde{c} = [\log_{10}(Re), \log_{10}(Sc)]
$$

并经过标准化线性放缩至 $[-1.0, 5.7]$ 区间作为条件投影网络（AdaLN）的输入。

---

## 3. HDF5 存储结构与规格

The Well 采用统一的 HDF5 层级群组格式存储多物理场数据。

### 3.1 目录结构树

```text
/
├── boundary_conditions/
│   ├── x_periodic/mask    (256,)   [bool] (全 True，X 方向周期性边界)
│   └── y_periodic/mask    (512,)   [bool] (全 True，Y 方向周期性边界)
├── dimensions/
│   ├── time               (200,)   [float64] (t: 0.0 ~ 19.9, dt = 0.1)
│   ├── x                  (256,)   [float32] (范围 [0.0, 1.0])
│   └── y                  (512,)   [float32] (范围 [0.0, 1.0])
├── scalars/
│   ├── Reynolds           ()       [float32]
│   └── Schmidt            ()       [float32]
├── t0_fields/                      (Rank-0 标量场)
│   ├── pressure           (N_traj, 200, 256, 512) [float32]
│   └── tracer             (N_traj, 200, 256, 512) [float32]
└── t1_fields/                      (Rank-1 向量场)
    └── velocity           (N_traj, 200, 256, 512, 2) [float32] (u_x, u_y)
```

### 3.2 维度与规格说明

| 属性 | 实际数值 | 说明 |
| :--- | :--- | :--- |
| **原始空间分辨率** | $256 \times 512$ | 均匀笛卡尔网格（$N_y \times N_x$） |
| **适配训练分辨率** | $128 \times 128$（快速训练）/ $256 \times 512$（高精训练） | 保持周期性边界插值下采样或全分辨率 |
| **时间步长 $\Delta t$** | $0.1$ 无量纲时间 | 相邻两帧间隔 $\Delta t = 0.1$ |
| **单轨迹总步数** | $T = 200$ | 覆盖 $t \in [0.0, 19.9]$ 充分演化过程 |
| **边界条件** | 双向严格周期边界（Periodic） | 网络使用 `padding_mode="circular"` |

---

## 4. 关键物理守恒性审计事实

在实测数据审计（[DATA_AUDIT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/DATA_AUDIT.md)）中确认了以下重要物理规律：

### 4.1 Schmidt 数不变性（示踪标量单向解耦）
* **原理**：示踪剂为被动标量，对流体 Navier-Stokes 方程无动量反作用力。
* **实测验证**：在相同初始流场和相同 $Re$ 下，比较 $Sc=0.1$ 与 $Sc=1.0$ 轨迹：
  - 速度场最大差异 $|u_A - u_B| < 1.19 \times 10^{-7}$，平均差异 $\approx 5.2 \times 10^{-12}$（达单精度浮点极限，物理严格一致）；
  - 示踪标量最大偏差为 $0.5386$（反映出质量扩散强度的显著物理差异）。

### 4.2 压力零均值规范（Gauge Freedom）
* 在不可压缩流中，只有压力梯度 $\nabla p$ 影响流场加速度，绝对压力值具有常数自由度 $\int_{\Omega} p \, d\Omega = 0$；
* 仿真初始步 $t=0$ 时压力严格为 0；在后续演化步中空间均值亦保持在 $10^{-7}$ 量级；
* **解码器约束**：解码器必须加入非就地压力均值消去算子 $p \leftarrow p - \frac{1}{|\Omega|}\sum p$，避免无物理意义的常数漂移。

---

## 5. 数据集划分与防动力学泄漏协议

### 5.1 官方划分的潜在泄漏陷阱
在 The Well 官方原始划分中，跨不同 $Sc$ 文件的部分初始轨迹编号存在**初始场完全相同**的现象（例如 `valid/...Schmidt_1e0.hdf5` 中的 Traj 0 与 `train/...Schmidt_1e-1.hdf5` 中的 Traj 26 流场完全重合）。若按传统多任务方式打乱，模型在验证集上预测速度场实际上会变成“记忆回放”。

### 5.2 我们的独立轨迹隔离方案
为确保评测具备 100% 泛化说服力：
1. **严格按初始扰动轨迹 ID 隔离**：验证集与测试集采用独立的初始扰动种子（Unseen Initial Conditions）；
2. **测试集独立评测**：测试集固定包含 36 条未知轨迹，涵盖多种 $Re$ 与 $Sc$ 组合；
3. **评测基准统一**：所有基线模型（Persistence、FNO-2D、PDE-Transformer、World Model）均基于相同的数据切片与窗口规则进行评估。

---

## 6. 代码接口与数据加载范式

核心数据加载逻辑封装于 [src/data/shear_flow_dataset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/shear_flow_dataset.py)。

### 6.1 典型调用代码

```python
from src.data.shear_flow_dataset import get_dataloader

# 构建自回归滚动训练 DataLoader
train_loader = get_dataloader(
    data_dir="/root/autodl-tmp/datasets/shear_flow",
    split="train",
    history_steps=4,      # L = 4 帧历史
    forecast_steps=2,     # H = 2 步滚动预测
    batch_size=4,
    spatial_res=(128, 128),
    shuffle=True,
    num_workers=4
)

for batch in train_loader:
    # x: [B, L, 4, 128, 128]  -> 历史物理场 [u, v, p, s]
    # y: [B, H, 4, 128, 128]  -> 真实未来物理场
    # cond: [B, 2]            -> 物理参数 [log10(Re), log10(Sc)]
    x = batch["history"].cuda()
    y = batch["future"].cuda()
    cond = batch["cond"].cuda()
    ...
```
