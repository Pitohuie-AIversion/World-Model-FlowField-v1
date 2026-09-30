# FM-R2 Rollout-Aware Multi-Seed Replication Protocol (Pre-Registration)

## 1. 核心假说与实验定位

本实验旨在严格检验基于自生成历史条件（Self-Generated-History Conditioning）的自回归流匹配微调机制在跨随机种子下的稳健性：

$$\boxed{\text{Robustness of the C2 vs R2-A fine-tuning treatment} \mid \text{Fixed Seed-42 Parent World Model}}$$

- **上游基础模型与参考基线**：保持固定的 Seed-42 预训练模型（冻结的 D0 Transformer 骨干与 Seed-42 Parent FM 检查点，以及 G0/G1 高斯基线）；
- **自变量（Treatment）**：
  - **C2（Control）**：Teacher-Forced 2-step 微调（第 2 步条件历史使用真实状态 $z_{t+1}$）；
  - **R2-A（Treatment）**：Rollout-Aware 2-step 微调（第 2 步条件历史使用自生成状态 $\hat{z}_{t+1} \sim \mathrm{FM}(\alpha_{\mathrm{train}}=0.5)$）。
- **因变量与效应量**：多步外推物理误差（VRMSE、散度、涡度）与径向能谱相对 L2 误差。

---

## 2. 种子分层与角色界定

依据先导研究与证实性实验的统计分离原则，实验种子被严格区隔：

1. **Discovery Phase（探索发现阶段）**：
   - **Seed 42**（已完成并正式封存）：确立了单步高保真建模以及长程 $h=10$ 误差与能谱保真度的显著改善假说。
2. **Confirmatory Replication Phase（证实性复现阶段）**：
   - **Seed 43, Seed 44**（核心 2 种子复现，形成最低 3 种子总体）；
   - **Seed 45, Seed 46**（算力许可下的可选扩展种子）。
3. **报告原则**：
   - 独立报告 Seed 42 发现结果；
   - 独立报告 Seeds 43–44（及后续）证实性复现结果；
   - 报告全种子的汇总描述性统计。

---

## 3. 预先指定的评测终点（Pre-Specified Endpoints）

实验冻结以下终点指标，严禁在观测到训练结果后事后调整核心指标优先级：

### 核心主要终点（Primary Outcomes）
1. **Primary 1 — $h=10$ Ensemble VRMSE**：
   - 衡量长期宏观均值场的外推预测精度；
   - Seed 42 探索基线效应：$-2.10\%$（$1.2822 \rightarrow 1.2553$）。
2. **Primary 2 — $h=10$ Ensemble Field Spectral Relative L2 Error**：
   - 衡量宏观集成场的大尺度能谱能量分布保真度；
   - Seed 42 探索基线效应：$-19.69\%$（$4.87\% \rightarrow 3.91\%$）。
3. **Primary 3 — $h=10$ Individual-Member Spectral Relative L2 Error**：
   - 衡量单体物理轨迹微观尺度谱保真度，排除单纯集成平均带来的平滑伪影；
   - Seed 42 探索基线效应：$-13.86\%$（$5.47\% \rightarrow 4.71\%$）。
4. **Primary 4 — $h=5$ Ensemble VRMSE**：
   - 严格检验短时程微幅精度代价（Short-horizon trade-off cost）假说是否跨种子稳定存在；
   - Seed 42 探索基线效应：$+0.42\%$（$0.6439 \rightarrow 0.6466$）。

### 次要终点（Secondary Outcomes）
- **Sample Mean VRMSE vs GT**（$h=5, 10$）
- **Sample RMS Divergence & Ratio vs GT**（$h=5, 10$）
- **Sample Vorticity RMSE vs GT**（$h=5, 10$）
- **One-Step Latent / Physical CRPS**（$K=32, \alpha=0.5$）
- **One-Step Pooled Spread-Skill Ratio (SSR)**

---

## 4. 实验不变量与控制变量契约

所有种子间的训练与评测执行完全一致的超参数配置，严禁在种子间做超参数微调：

- **微调轮数**：`epochs = 1`
- **学习率与权重衰减**：`lr = 2e-4, weight_decay = 1e-4, grad_clip = 1.0`
- **批次大小**：`batch_size = 16`（每个 epoch 52 次优化器更新）
- **ODE 求解器**：`solver = midpoint, num_flow_steps = 10`
- **噪声尺度**：`sample_noise_scale = 0.5`（训练与评测严格对齐：$\alpha_{\mathrm{train}} = \alpha_{\mathrm{eval}} = 0.5$）
- **加密校验绑定（Fail-Closed）**：
  - D0 Checkpoint SHA: `edddbe8a2528f848975ed33921bb8d2df13059cedb149c36f8132e20b255c0f6`
  - Parent FM Checkpoint SHA: `0a2ac5dc8af8e9f6d532d8e4c0452a9e679f238d4e6470d323bd21b2c473c7ec`（`expected_parent_seed = 42`）
  - Data Split Hash: `41fbe6ebe7edd460b4353fd6cf20ad064f1222fdcb4558b04ff1a50be390b93d`
  - Normalizer Hash: `3a0fe52689657618a92a90881aca42d349639502e956017c6fcbf0368c5d4bec`
  - Residual Stats SHA: `448b2c338d7ec9e3eb05c1fb1ca4d0ca2088eae2e7e6bec5fd2fab0c27978f36`

---

## 5. 硬件分配与设备效应平衡（GPU Alternation）

为消除固定显卡硬件或底层计算流可能引入的系统性设备偏差（Device-Specific Bias），在种子间严格轮换 GPU 分配：

| 实验批次 | 分支名称 | 训练种子 | 指定硬件设备 | 输出目录 |
|---|---|:---:|:---:|---|
| **Seed 43** | C2 (Control) | 43 | `cuda:0` | `outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/43/C2` |
| | R2-A (Treatment) | 43 | `cuda:1` | `outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/43/R2_A` |
| **Seed 44** | C2 (Control) | 44 | **`cuda:1`** | `outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/44/C2` |
| | R2-A (Treatment) | 44 | **`cuda:0`** | `outputs/checkpoints/probabilistic/flow_matching_r2_alpha05_seeds/44/R2_A` |

---

## 6. 统计推断与层级分析协议（Cluster-Aware Inference）

### 1. 统计分析单元界定
严禁将验证集中的 144 个重叠滑动窗口当成 144 个独立样本进行统计假设检验。数据具有严格的层级结构：
$$\boxed{\text{Training Seed } s \rightarrow \text{Physical Trajectory } j \in \{1,\dots,6\} \rightarrow \text{Overlapping Windows (轨迹内聚合)}}$$

### 2. 成对效应差分计算
对每个训练种子 $s$ 与物理验证轨迹 $j$，计算配对差异：
$$\Delta_{s, j} = M_{s, j}^{R2A} - M_{s, j}^{C2}$$
- 3 种子设计（Seeds 42, 43, 44）共计产生 $3 \times 6 = 18$ 个轨迹级配对观测值；
- 5 种子设计（Seeds 42–46）共计产生 $5 \times 6 = 30$ 个轨迹级配对观测值。

### 3. 统计推断与验收标准
- **Within-Seed 相关性控制**：同一 seed 下的 6 条轨迹共享同一训练权重，在估计标准误时采用聚类稳健标准误（Cluster-Robust SE）；
- **科学结论成立判据**：
  - 核心要求是总体期望效应方向为负：$E[\Delta_{h10}] < 0$ 且置信区间支持优势方向；
  - 径向能谱 L2 相对误差效应量稳定处于 $-10\% \sim -20\%$ 的显著区间；
  - 短程 $h=5$ 上保持小幅度或可控的局部精度代价（证实 trade-off 特征）；
  - 不强求所有种子、所有轨迹呈现机械化的 100% 严格同号，以效应量分布与层级均值为最终发表依据。
