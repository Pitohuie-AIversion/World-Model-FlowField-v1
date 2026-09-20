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
