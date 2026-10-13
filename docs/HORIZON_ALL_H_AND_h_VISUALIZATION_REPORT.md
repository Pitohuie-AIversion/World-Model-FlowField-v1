# 流场世界模型：全跨度小 $h$ 与大 $H$ 预测场对比报告 (GT / Pred / Error)

> **任务目标**：将所有具备可视化条件的训练时序跨度（大 $H \in \{1, 2, 4, 8, 12, 16\}$）与推演步长（小 $h \in \{1, 2, 4, 8, 12, 16, 20, 30\}$），统一按照真实场（Ground Truth, GT）、模型预测场（Prediction, Pred / prd）以及绝对误差场（Absolute Error, $|prd - GT|$）进行全量可视化与系统性评估。  
> **基准流场**：The Well 剪切流测试集锚定样本（Sample #0: $Re=10^4, Sc=0.10$）。  
> **图件产物目录**：`outputs/figures/horizon_all_h_and_H/`

---

## 一、 核心物理与工程结论

1. **分布内（$h \le H$）拟合保真度**：
   - 当推演步长处于模型的训练视界以内时，所有大 $H$ 模型均保持极高拟合精度；
   - 随训练视界由 $H=1$ 稳步提升至 $H=16$，网络在时序推演前期的多步累积误差被显式惩罚，使得单步输出更加平滑且符合长程流体动力学约束。

2. **分布外（$h > H$）自回归推演退化机制**：
   - **$H=1$（E0 单步基线）**：缺乏多步展开监督，在自回归推进至 $t=4$ 时开始积累高频伪能量，至 $t=16 \sim 30$ 时出现不可逆的高频网格色散与能量发散；
   - **$H=2$（短程滚动 E4-H2）**：相较于 $H=1$ 极大抑制了初期色散，但在推进至 $t=8 \sim 12$ 时开始出现涡量撕裂；
   - **$H=4$（中程课程式 E4-H4）**：在 $t \le 8$ 保持稳定，在极远步长 $t=30$ 时剪切层发生相位滞后；
   - **$H=8$ 与 $H=12$**：表现出显著优异的中长程稳定性，涡核配对形态在 $t=16 \sim 20$ 依然清晰可辨；
   - **$H=16$（长程长训 Long-Best）**：在 $t \in [1, 16]$ 的全局范围内保持低误差，并在极限步长 $t=30$ 下依然有效压制了色散爆炸，展现出最鲁棒的长程吸引子结构。

3. **物理变量敏感性分异**：
   - **水平速度 $u$**：属于大尺度能量主导变量，随着 $H$ 增加，长程误差主要表现为剪切带厚度的轻微扩散与相位微移；
   - **涡量 $\omega$**：对高频空间导数极其敏感，是检验模型是否发散的关键“试金石”。$H1$ 与 $H2$ 在后期的主要破裂表现即为伪涡丝爆发，而 $H8 \sim H16$ 则保持物理涡核的完整性；
   - **示踪物 $s$**：受对流扩散主导，长程推演中大 $H$ 模型能有效约束示踪物锋面的尖锐性，避免过度扩散渗漏。

---

## 二、 定量误差热力图：小 $h$ vs 大 $H$ 演化大盘

![小 h 与大 H 全局 VRMSE 误差演化热力图与推演退化曲线](../outputs/figures/horizon_all_h_and_H/horizon_quantitative_heatmap_sample_0.png)

- **左图（推演退化曲线）**：各模型在 $h \in [1, 30]$ 的全长 VRMSE 轨迹；
- **右图（二维误差热力图）**：横轴为推演步长 $h \in [1, 30]$，纵轴为模型训练视界 $H \in \{1, 2, 4, 8, 12, 16\}$，蓝色竖线标示出各模型的训练视界 $H$ 截断点。可以清晰观察到“视界截断后误差加速增长”的典型动力学分水岭。

---

## 三、 全大 $H$ × 小 $h$ 横向对比综合全景图 (Grand Horizon Ladder)

各图均按照 8 个关键时间步（$t \in \{1, 2, 4, 8, 12, 16, 20, 30\}$）垂直排列，水平展开 13 列（1 列 GT + 6 组 [Pred, Error]）：

### 1. 水平速度场 $u$ 横向对比全景图
![水平速度 u 全大 H 与小 h 横向对比全景大图](../outputs/figures/horizon_all_h_and_H/grand_horizon_ladder_u_sample_0.png)

### 2. 涡量场 $\omega$ 横向对比全景图
![涡量场 omega 全大 H 与小 h 横向对比全景大图](../outputs/figures/horizon_all_h_and_H/grand_horizon_ladder_vorticity_sample_0.png)

### 3. 被动标量示踪物 $s$ 横向对比全景图
![被动标量示踪物 s 全大 H 与小 h 横向对比全景大图](../outputs/figures/horizon_all_h_and_H/grand_horizon_ladder_tracer_sample_0.png)

---

## 四、 单模型时序展开面板（按大 $H$ 逐个呈现：GT / Pred / Error）

每个大 $H$ 模型独立生成一张完整时序展开图（第一行：GT；第二行：Pred；第三行：|Pred - GT|）：

### 1. $H=1$ (E0 单步纯物理网格基线)
- 水平速度 $u$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H1_u_sample_0.png`
- 涡量场 $\omega$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H1_vorticity_sample_0.png`
- 示踪物 $s$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H1_tracer_sample_0.png`

### 2. $H=2$ (E4-H2 短程多步滚动基准)
- 水平速度 $u$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H2_u_sample_0.png`
- 涡量场 $\omega$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H2_vorticity_sample_0.png`
- 示踪物 $s$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H2_tracer_sample_0.png`

### 3. $H=4$ (E4-H4 课程式递进模型)
- 水平速度 $u$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H4_u_sample_0.png`
- 涡量场 $\omega$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H4_vorticity_sample_0.png`
- 示踪物 $s$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H4_tracer_sample_0.png`

### 4. $H=8$ (E4-H8 长程最优检查点)
- 水平速度 $u$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H8_u_sample_0.png`
- 涡量场 $\omega$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H8_vorticity_sample_0.png`
- 示踪物 $s$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H8_tracer_sample_0.png`

### 5. $H=12$ (E4-H12 超长跨度长程最优检查点)
- 水平速度 $u$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H12_u_sample_0.png`
- 涡量场 $\omega$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H12_vorticity_sample_0.png`
- 示踪物 $s$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H12_tracer_sample_0.png`

### 6. $H=16$ (E4-H16 极限跨度长程最优检查点)
- 水平速度 $u$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H16_u_sample_0.png`
- 涡量场 $\omega$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H16_vorticity_sample_0.png`
- 示踪物 $s$：`outputs/figures/horizon_all_h_and_H/rollout_evolution_H16_tracer_sample_0.png`

---

## 五、 资产与契约清单

所有 22 幅高分辨率图件及配套的元数据 JSON 已完整落盘保存在：
`outputs/figures/horizon_all_h_and_H/`
包含每个模型的权重路径、完整 SHA-256 校验和、Git Commit 锚定信息以及具体数值统计。
