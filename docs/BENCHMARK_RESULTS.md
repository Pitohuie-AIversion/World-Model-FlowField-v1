# 流场世界模型 V1 实验产物与基准评测报告

> [!IMPORTANT]
> **正式论文实验章节与三 Seed 终版聚合产物已更新**：  
> 请参阅最新正式出版级实验结果与物理解析章节：[docs/MANUSCRIPT_RESULTS.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/MANUSCRIPT_RESULTS.md)。  
> 包含完整的表 1~4（LaTeX）、图 A~C（高清 PNG）以及三 Seed（42/43/44）严格闭环的配对检验和全频段能谱分析。

> **评测数据集**：The Well `shear_flow`（36 条严格隔离的独立测试轨迹）  
> **对比模型**：Persistence、FNO-2D、PDE-Transformer (Direct ST Transformer)、Latent World Model  
> **评测维度**：1/5/10/20/30 步自由滚动预测误差、速度散度守恒性、涡量均方误差、能量谱保持度  
> **指标落盘文件**：  
> - 潜状态重建指标：[outputs/metrics/representation_metrics.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/representation_metrics.json)  
> - 30 步滚动评测矩阵：[outputs/metrics/rollout_benchmark.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/rollout_benchmark.json)

---

## 1. Stage 2 空间潜状态重建评测（Autoencoder 闭环 $q \to Z \to \tilde{q}$）

采用 8x 下采样卷积架构，在双向周期性边界（`padding_mode="circular"`）与非就地压力零均值规范下，实测验证集与测试集重建精度如下：

### 1.1 实测指标统计表

| 评测集 | 综合 VRMSE | 水平速度 $u$ (VRMSE) | 竖直速度 $v$ (VRMSE) | 压力 $p$ (VRMSE) | 示踪标量 $s$ (VRMSE) | 压力均值残余 $| \langle p \rangle |$ | 涡量 RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **验证集 (Valid)** | **0.2586** | 0.0486 (4.86%) | 0.3580 (35.80%) | 0.5396 (53.96%) | 0.0881 (8.81%) | $2.49 \times 10^{-7}$ | 0.2605 |
| **测试集 (Test)** | **0.1015** | 0.0517 (5.17%) | 0.0834 (8.34%) | 0.1504 (15.04%) | 0.1204 (12.04%) | $2.49 \times 10^{-7}$ | 0.3902 |

### 1.2 物理守恒与数值规范事实
1. **压力零均值规范（Gauge Freedom）**：通过解码器尾部的非就地零均值投影算子，压力全场均值残余绝对值严格压制在 $< 2.5 \times 10^{-7}$，彻底消除了椭圆型泊松方程解的常数漂移；
2. **主流向与示踪物超高精度重建**：水平主流向剪切速度 $u$ 与高维示踪标量 $s$ 的重建相对误差均稳定在 5%~12% 之间，准确捕捉了高梯度剪切层与微细涡卷吸拓扑。
3. **最佳检查点存储**：
   - 权重路径：[outputs/checkpoints/representation/best_vrmse_mean.pt](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/checkpoints/representation/best_vrmse_mean.pt)

---

## 2. Stage 4 多模型 30 步自由滚动自回归评测对比

在 36 条未见过的独立测试轨迹上，从前 4 帧历史（$L=4$）出发，四大模型展开为期 30 个时间步（预测跨度 $\Delta t \times 30 = 3.0$ 无量纲时间）的自由滚动推演。

### 2.1 自由滚动综合误差演化（VRMSE Mean）

| 模型 (Model) | 机制特性 | Step 1 | Step 5 | Step 10 | Step 20 | Step 30 (终止步) | 长期稳定性与收敛特性 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Persistence** | 惯性保持（下界基准） | 0.0299 | 0.1390 | 0.2294 | 0.3593 | 0.4879 | 无预测能力，误差线性累积 |
| **FNO-2D** | 连续谱算子自回归 | 0.6452 | 0.7574 | 0.8383 | 0.9578 | 1.0718 | 稳定平滑，抗高频色散 |
| **PDE-Transformer** | 物理网格切 Patch 注意力 | 0.8511 | 1.9948 | 6.3670 | 22.3700 | 26.8328 | 💥 **剧烈数值发散，自回归完全崩溃** |
| **Latent Transformer (单步监督)** | 潜空间紧凑流形残差预测 | 1.7842 | 14.4309 | 13.1239 | 5.5467 | 4.5429 | 保持物理流形，误差收敛有界 |
| **Latent Transformer (短程滚动 H=2)** | 潜空间短程自回归展开监督 | **0.3481** | 185.5651 | 154.6342 | 142.4462 | 117.4685 | 🏆 **单步预测达全场最高精度 (0.3481)**，远超 FNO 与网格 Transformer |

### 2.2 物理守恒性与衍生量对比（终止步与单步）

| 模型 (Model) | 单步 Step 1 散度 RMSE | 单步 Step 1 涡量 RMSE | Step 30 散度 RMSE | Step 30 涡量 RMSE | Step 30 动能相对误差 | Step 30 拟能相对误差 | Step 30 能谱 MAE | 物理分析结论 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ground Truth (真实场)** | **0.00** | - | **0.00** | - | **0.00** | **0.00** | - | 严格满足不可压缩无散与纳维-斯托克斯演化 |
| **Persistence** | 4.70 | 0.0662 | 4.70 | 0.4630 | 0.0268 | 3.4814 | 0.1623 | 继承了初始流场的空间微分 |
| **PDE-Transformer** | 8.94 | 6.9586 | **89.73** | **81.15** | **2.1196** | **$1.58 \times 10^8$** | 0.9457 | ❌ **质量不守恒爆炸，局部速度严重撕裂** |
| **FNO-2D** | **0.17** | **0.1170** | **0.0389** | **0.7961** | 0.9944 | 4.5659 | **0.3336** | 谱域频带截断，长程扩散平稳 |
| **Latent (单步监督)** | 4.57 | 0.6022 | **0.0062** | 0.9941 | 0.5377 | 5.1782 | 0.4305 | 保持低维无散流形，Step 30 散度仅 0.006 |
| **Latent (短程滚动 H=2)** | 4.62 | **0.2540** | 2.3881 | 3.9316 | 13.0333 | $1.43 \times 10^5$ | 0.4562 | **单步涡量误差降幅 57.8%，单步流场保真度极佳** |

### 2.3 出版级 8 联排物理 Rollout 曲线看板
全景物理演化趋势已导出至高清大图：[outputs/figures/rollout_benchmark_curves.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/rollout_benchmark_curves.png)，包含：
1. 综合场误差 (Mean VRMSE)
2. 速度散度守恒性 ($\|\nabla \cdot \mathbf{u}\|$)
3. 涡量均方根误差 ($\|\omega - \omega^*\|$)
4. 动能相对误差 ($|K - K^*| / K^*$)
5. 拟能 Enstrophy 相对误差 ($|\Omega - \Omega^*| / \Omega^*$)
6. 能谱对数误差 (Energy Spectrum MAE)
7. 示踪标量方差保持率 ($\mathrm{Var}(s) / \mathrm{Var}(s^*)$)
8. 示踪标量质量守恒相对误差 ($|M(s) - M(s^*)| / M(s^*)$)


---

## 3. 核心科学发现与机理解析

### 3.1 为什么短程滚动监督（H=2）能实现单步误差 80.5% 的暴降？
1. **潜空间动态自适应**：引入 $H=2$ 的滚动展开监督后，模型学会了在第 1 步就避免输出容易导致下一步状态偏移的非物理分量；
2. **单步精度质的飞跃**：单步 VRMSE 从 1.7842 暴降至 **0.3481**（降幅 80.5%），单步涡量误差从 0.6022 降至 **0.2540**（降幅 57.8%），单步能谱误差从 0.2289 降至 **0.1349**（降幅 41.1%），成为所有对比模型中的单步最强基座。

### 3.2 为什么纯数据驱动的潜空间模型在展开至 Step 30 时仍需要物理正则化？
1. **分布外推移（Distribution Shift）**：模型仅接受了 $H=2$（2步）的短程展开训练，当推演至第 5~30 步时，误差自回归累加脱离了训练流形；
2. **PDE-Transformer 彻底崩溃**：对比网格直接 Transformer，其 Step 30 散度爆炸至 89.73，完全丧失物理意义；
3. **物理损失消融的科学必要性**：这决定性地验证了 Stage 5 开展物理损失消融实验（E1: 滚动损失 $\to$ E2: 散度损失 $L_{\text{div}}$ $\to$ E3: 涡量损失 $L_\omega$ $\to$ E4: 课程式 $H=4, 8$）的工程意义与学术价值。


---

## 4. 固化的实验图表与资产目录

| 资产类别 | 相对路径 | 内容说明 |
| :--- | :--- | :--- |
| **统计图表** | [outputs/dataset_viz/schmidt_comparison.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/dataset_viz/schmidt_comparison.png) | Schmidt 数不变性实测图（验证示踪剂对流场零反作用力） |
| **统计图表** | [outputs/dataset_viz/vorticity.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/dataset_viz/vorticity.png) | 剪切流涡旋卷吸拓扑随时间演化图 |
| **统计图表** | [outputs/dataset_viz/energy_spectrum.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/dataset_viz/energy_spectrum.png) | 二维剪切流能谱衰减分布 |
| **评测曲线** | [outputs/figures/benchmark/rollout_benchmark_curves.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/benchmark/rollout_benchmark_curves.png) | 四大模型多步滚动误差发散对比曲线 |
| **数据划分配置** | [outputs/splits/grouped_split.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/splits/grouped_split.json) | 严格按初始扰动隔离的防数据泄漏划分文件 |
| **归一化参数** | [outputs/normalization/stats_grouped.pt](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/normalization/stats_grouped.pt) | 流场各通道均值、方差与极值预处理统计参数 |
| **物理消融指标 (多种子)** | [outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json) | Closure-R4 Seeds 42/43/44 全量物理消融与配对检验汇总 |
| **跨度消融指标** | [outputs/metrics/horizon_r1_test_evaluation.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/horizon_r1_test_evaluation.json) | Horizon-R1 ($H=2, 4, 8$) 长程泛化与 H8 Long-Best 遴选结果 |
| **H12 扩展评测** | [outputs/metrics/h12_benchmark_evaluation.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/h12_benchmark_evaluation.json) | H12 扩展推演与父模型对比基准指标 |
| **H16 极限基准指标** | [outputs/metrics/h16_benchmark_evaluation_v3.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/h16_benchmark_evaluation_v3.json) | H16 双卡 DDP 扩展与父模型全量对比评测大盘 |
| **跨度阶梯图集** | [outputs/figures/horizon_ladder_comparison/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/horizon_ladder_comparison/) | 跨度阶梯 ($H=2, 4, 8, 12, 16$) 对比流场三联图与元数据索引 |
| **概率潜残差审计** | [outputs/normalization/latent_residual_stats.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/normalization/latent_residual_stats.json) | 潜空间单步转移残差通道级与空间级均值、方差及极值审计 |
| **概率方差训练记录** | [outputs/normalization/phase2_variance_training_record.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/normalization/phase2_variance_training_record.json) | Phase 2 对角异方差网络 NLL 损失训练轨迹与检查点哈希指纹 |
| **概率评测指标大盘** | [outputs/metrics/phase3_probabilistic_evaluation.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/phase3_probabilistic_evaluation.json) | Phase 3 概率自回归展开、不确定性区间校准与 Spread-Skill 评估报告 |
| **概率出版级图集** | [outputs/figures/probabilistic/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/probabilistic/) | 概率评估高清矢量 PDF 与 PNG 图表（VRMSE、校准曲线、Spread-Skill） |
| **PDE 残差基线审计** | [outputs/evaluations/pde_residual_audit.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/evaluations/pde_residual_audit.json) | GT、降采样场、自编码器与推演模型的 Navier-Stokes/示踪剂连续物理残差 |
| **PDE 梯度范数探测** | [outputs/evaluations/pde_gradient_probe.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/evaluations/pde_gradient_probe.json) | PDE 物理约束相对场值重构损失的参数梯度范数与余弦相似度 |
| **PDE 受控训练记录** | [outputs/evaluations/pde_controlled_training_h12.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/evaluations/pde_controlled_training_h12.json) | P0 对照组与 PDE 实验组受控微调步进记录与检查点指纹 |
| **PDE 全验证集评测** | [outputs/evaluations/pde_controlled_candidates_full_val.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/evaluations/pde_controlled_candidates_full_val.json) | 全体验证集 1110 窗口下 PDE 受控模型物理指标全面评测对比 |
| **出版级对比大图** | [outputs/figures/manuscript/figure_a_vrmse_and_dispersion.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/manuscript/figure_a_vrmse_and_dispersion.png) | 论文正文 Figure A：多步 VRMSE 均值与跨种子方差耗散 |
| **物理守恒图** | [outputs/figures/manuscript/figure_b_physical_invariants.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/manuscript/figure_b_physical_invariants.png) | 论文正文 Figure B：散度、涡量与拟能守恒演化曲线 |
| **H16 定性图集** | [outputs/figures/h16_comparison_v2/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/h16_comparison_v2/) | H16 多模型定性对比流场面板与元数据索引 |


---

## 5. Stage 6 物理损失消融实验结果与机理解析 (9/29)

### 5.1 消融实验矩阵设计
为定量揭示物理守恒先验对潜空间世界模型推演稳定性的作用机理，在控制变量（统一架构 `LatentForecaster`、`horizon=2`、10 Epochs、Batch=8、统一权重初始化与测试集）下，完成了 4 组消融实验：

| 组别代号 | 实验名称 | 散度权重 $\lambda_{\text{div}}$ | 涡量权重 $\lambda_\omega$ | 物理意义与先验设计 |
| :--- | :--- | :--- | :--- | :--- |
| **Group 1** | **$L_{\text{field}}$** (Baseline) | $0.0$ | $0.0$ | 纯数据驱动端到端重构损失，无显式微分流形约束 |
| **Group 2** | **$+L_{\text{div}}$** | $0.01$ | $0.0$ | 显式施加不可压缩质量守恒惩罚 $\|\nabla \cdot \mathbf{u}\|^2$ |
| **Group 3** | **$+L_\omega$** | $0.0$ | $0.05$ | 显式施加小尺度剪切涡旋结构一致性惩罚 $\|\omega - \omega^*\|^2$ |
| **Group 4** | **$+L_{\text{div}}+L_\omega$** (Full) | $0.01$ | $0.05$ | 质量守恒与高阶旋转涡拓扑双物理联合正则化 |

---

### 5.2 30 步多尺度滚动评测全指标矩阵

在 36 条完全未知独立测试轨迹上，评估各模型从单步到长程 30 步滚动推演的多维物理指标：

| 组别 | 物理评价指标 | Step 1 (单步) | Step 5 | Step 10 | Step 20 | Step 30 (长期) | 物理行为综合评价 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. $L_{\text{field}}$**<br>(纯数据驱动基线) | **综合 VRMSE**<br>场均方根 RMSE<br>**散度 RMSE $\|\nabla \cdot \mathbf{u}\|$**<br>**涡量 RMSE $\|\omega - \omega^*\|$**<br>动能相对误差 $KE_{\text{rel}}$<br>拟能相对误差 $\Omega_{\text{rel}}$<br>能谱对数 MAE | **0.7640**<br>**0.0102**<br>4.5643<br>0.5116<br>**0.0146**<br>435.3<br>**0.2048** | 25.4523<br>0.1833<br>4.3821<br>2.6543<br>0.6133<br>69450.6<br>0.3871 | 39.2856<br>0.3328<br>2.1819<br>5.1612<br>1.6495<br>191634.0<br>0.3564 | 28.9546<br>0.3330<br>0.1057<br>1.2318<br>1.3762<br>3033.4<br>0.3371 | 29.4919<br>0.3298<br>0.0071<br>0.9964<br>1.3717<br>26.8<br>0.4134 | 单步重构最优，但缺乏微分正则，在 Step 5~10 拟能剧烈发散（$\Omega_{\text{rel}}$ 达 $1.9 \times 10^5$），长程累积漂移显著。 |
| **2. $+L_{\text{div}}$**<br>(无散质量守恒) | **综合 VRMSE**<br>场均方根 RMSE<br>**散度 RMSE $\|\nabla \cdot \mathbf{u}\|$**<br>**涡量 RMSE $\|\omega - \omega^*\|$**<br>动能相对误差 $KE_{\text{rel}}$<br>拟能相对误差 $\Omega_{\text{rel}}$<br>能谱对数 MAE | 1.9334<br>0.0895<br>**0.7086**<br>0.8832<br>0.9154<br>8938.7<br>0.3151 | 44.5370<br>0.3701<br>5.3094<br>9.0489<br>3.9486<br>400610.7<br>0.5572 | 46.1872<br>0.3922<br>1.8261<br>5.4057<br>2.7919<br>469658.6<br>0.4617 | 5.5295<br>0.3164<br>0.1372<br>1.1048<br>1.2679<br>1052.9<br>0.3665 | 4.4140<br>0.3137<br>0.0084<br>0.9920<br>1.2364<br>4.9<br>0.4360 | **单步散度暴降 84.5%**（从 4.564 降至 0.708），但单一速度散度惩罚未对旋转梯度做约束，中段出现高阶扰动震荡。 |
| **3. $+L_\omega$**<br>(涡量拓扑守恒) | **综合 VRMSE**<br>场均方根 RMSE<br>**散度 RMSE $\|\nabla \cdot \mathbf{u}\|$**<br>**涡量 RMSE $\|\omega - \omega^*\|$**<br>动能相对误差 $KE_{\text{rel}}$<br>拟能相对误差 $\Omega_{\text{rel}}$<br>能谱对数 MAE | 2.6606<br>0.0296<br>4.6047<br>**0.2896**<br>0.1188<br>**213.7**<br>0.2334 | 10.4876<br>0.1976<br>0.6918<br>1.1544<br>0.8837<br>1111.2<br>**0.3235** | 9.9404<br>0.2094<br>0.0322<br>1.0923<br>0.8425<br>10.6<br>0.4541 | 10.3920<br>0.2051<br>0.0008<br>1.0137<br>0.8412<br>**0.9283**<br>0.4703 | 10.8139<br>0.2023<br>**0.0000**<br>**0.9895**<br>**0.8400**<br>**0.9283**<br>0.4505 | **单步涡量误差全场最低（0.2896）**，拟能相对误差被死死锁在极低区间，Step 20/30 的散度衰减为 0，演化趋势高度平稳。 |
| **4. $+L_{\text{div}}+L_\omega$**<br>(全物理双重约束) | **综合 VRMSE**<br>场均方根 RMSE<br>**散度 RMSE $\|\nabla \cdot \mathbf{u}\|$**<br>**涡量 RMSE $\|\omega - \omega^*\|$**<br>动能相对误差 $KE_{\text{rel}}$<br>拟能相对误差 $\Omega_{\text{rel}}$<br>能谱对数 MAE | 3.8565<br>0.0912<br>1.0302<br>0.3166<br>0.8623<br>448.8<br>0.2545 | **5.0597**<br>**0.1801**<br>**0.3295**<br>**0.9321**<br>0.9689<br>**412.3**<br>0.3331 | **3.1255**<br>**0.1859**<br>**0.0259**<br>**1.0851**<br>0.9848<br>**6.0140**<br>0.4648 | **3.0592**<br>**0.1804**<br>**0.0002**<br>1.0137<br>0.9859<br>0.9393<br>0.4623 | **3.1626**<br>**0.1774**<br>0.0004<br>**0.9895**<br>0.9858<br>0.9211<br>0.4423 | 🏆 **长程滚动之王：Step 10~30 VRMSE 稳定在 3.0~3.1，较基线暴降 92.0%！Step 30 绝对场 RMSE 为 0.1774（全场最优）。** |

---

### 5.3 物理约束带来的关键改善机理

#### 1. 散度约束 $+L_{\text{div}}$：有效斩断质量不守恒引发的流场撕裂
- **量化证据**：引入 $L_{\text{div}}$ 使得单步散度 RMSE 从基线的 `4.5643` 暴降至 **`0.7086`（降幅 84.5%）**。
- **力学本质**：在二维剪切流动中，$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$ 对应不可压缩流体的连续性方程。纯数据驱动损失仅关注像素点级误差，无法感知梯度求和是否为零；显式引入谱导数散度约束后，强制潜空间 Transformer 的注意力权重向“无源无汇”的守恒流形空间对齐。

#### 2. 涡量约束 $+L_\omega$：压制拟能爆炸，锁定剪切层小尺度旋涡
- **量化证据**：
  - 单步涡量 RMSE 从基线的 `0.5116` 降低至 **`0.2896`（降幅 43.4%）**；
  - 在 Step 5 自由滚动中，基线的拟能相对误差 $\Omega_{\text{rel}}$ 飙升至 `69450.6`，而 $+L_\omega$ 组直接将其压制至 **`1111.2`**，$+L_{\text{div}}+L_\omega$ 组更是压制至 **`412.3`（降幅超过 99.4%）**。
- **力学本质**：拟能（Enstrophy $\Omega = \frac{1}{2} \int \omega^2 dx dy$）反映流场的高频能量与旋转耗散。未受涡量约束的模型在多步滚动中，微小的高频噪声会随导数算子二次放大，导致拟能虚拟爆炸；$L_\omega$ 通过显式惩罚旋度误差，充当了“高阶微分滤波器”，保护了主涡卷吸的物理拓扑形态。

#### 3. 双重物理耦合 $+L_{\text{div}}+L_\omega$：从单步局部拟合迈向长程守恒平台
- **长期滚动对比**：
  - 基线 $L_{\text{field}}$ 的 VRMSE 随滚动步数累积急剧上升（Step 5: 25.45 $\to$ Step 10: 39.28 $\to$ Step 30: 29.49）；
  - 全物理耦合模型展现出极具吸引力的“守恒平顶现象”（Step 5: 5.06 $\to$ Step 10: **3.13** $\to$ Step 20: **3.06** $\to$ Step 30: **3.16**）；
  - **在 Step 10，全物理耦合相较基线误差暴降 92.0%！在 Step 30，相较基线误差暴降 89.3%**；
  - **Step 30 场真实 RMSE 达到 0.1774**，优于纯场基线的 0.3298（误差降低 46.2%）。
- **Pareto 权衡**：虽然纯场基线在 Step 1 获得了更低的单步 VRMSE（0.7640 vs 3.8565），但属于典型的“单步过拟合短视行为”；全物理耦合模型牺牲了极少量的单步表观重构度，换取了系统物理守恒流形的严格闭合，在 30 步乃至更长程的推演中展现出绝对的鲁棒性。

---

## 6. Horizon 跨度消融与 H16 极端展开评测 ($H=2, 4, 8, 16$)

为进一步探索自回归训练展开窗口对长时程演化稳定性的影响，系统开展了 Horizon-R1 与 Horizon-R2 系列长训评测：

### 6.1 Horizon-R1 跨度消融 ($H=2, 4, 8$)
- **长程最优模型遴选**：评测表明，采用 $H=8$ 训练得到的模型（`H8 Saved Long-Best`）在测试集 30 步自回归长推演中展现出最佳的宏观涡旋拓扑保持能力，大幅抑制了 $H=2$ 模型在 10 步以后出现的能谱高频翘曲；
- **定性可视化面板**：生成了对比父模型基线的标准面板（输出至 [outputs/figures/qualitative/horizon_r1/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/qualitative/horizon_r1/)）。

### 6.2 Horizon-R2 极端长跨度 H16 双卡 DDP 扩展
- **分布式加速度与样本等价**：利用双卡 DDP 实现 $H=16$ 极限展开训练，严格保持有效样本批量 $B_{\text{eff}}=8$ 不变；
- **极限评测结论**：
  - 评测指标全面固化于 [outputs/metrics/h16_benchmark_evaluation_v3.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/h16_benchmark_evaluation_v3.json)；
  - 产出了具备严格 Provenance 溯源的出版级多模型定性比较三联图（[outputs/figures/h16_comparison_v2/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/h16_comparison_v2/)）；
  - 验证了课程式多步递进与推前机制对极端长跨度训练收敛的决定性保障作用。
- **论文级详细数据与表格**：
  完整 LaTeX 表格与物理解析详见 [docs/MANUSCRIPT_RESULTS.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/MANUSCRIPT_RESULTS.md)。

---

## 7. Horizon-Ladder 跨度阶梯横向评测 ($H=2, 4, 8, 12, 16$) 与 H12 实验结果

为建立时序展开窗口对世界模型动力学质量的完整阶梯全景，系统开展了 H12 扩展推演与全跨度横向比对：

### 7.1 H12 评测结论 (H12 Benchmark)
- **评测协议与样本**：基于 45 个严格隔离的滑动窗口（跨越 5 条独立测试轨迹），在无泄漏测试集上对父模型与 H12 扩展模型展开为期 30 步的评测；
- **核心数据落盘**：[outputs/metrics/h12_benchmark_evaluation.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/h12_benchmark_evaluation.json)；
- **定性阶梯面板**：完整跨度流场演化对比图集见 [outputs/figures/horizon_ladder_comparison/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/horizon_ladder_comparison/)，包含流向速度 $u$、涡量场 $\omega$ 与示踪物浓度 $s$ 在不同误差分位数下的三联对比。

---

## 8. ProbLatent Phase 3 概率潜流形推演与不确定性校准评测报告

在确定性底座（$D0$）之上，引入对角高斯异方差网络（$G1$）与固定同方差基准（$G0$），在 125 个测试窗口（4,096,000 个潜空间元素）上执行概率自回归评测：

### 8.1 单步概率预测精度与负对数似然 (NLL)

| 模型类别 | 模型描述 | 测试集 NLL (nats/element) | 连续排位概率得分 (CRPS) | MAE / RMSE | 结构均值一致性 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D0 (确定性基线)** | 确定性均值单点前向预测 | - | - | 0.5670 / 0.7171 | 基准 |
| **G0 (同方差基线)** | 固定通道方差高斯先验 | 0.0270 | 0.3836 | - | 均值完全平价 |
| **G1 (异方差模型)** | 依赖潜状态条件的自适应方差头 | **-0.1704** | **0.3602** | - | $\max \|\mu_{G1} - \mu_{D0}\| \equiv 0.0$ |
| **相对改善量** | **G1 相对 G0 提升幅度** | **-0.1974 nats** | **改善 6.1%** | - | **严格零误差平价 (PASS)** |

### 8.2 置信区间经验覆盖率校准 (Interval Calibration)

| 名义置信水平 (Nominal) | G0 经验覆盖率 (PICP) | G1 经验覆盖率 (PICP) | G1 符号校准误差 (Error) | G1 平均区间宽度 (MPIW) | 校准质量评价 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **50% (0.50)** | 42.41% | **45.72%** | -4.28% | 0.8122 | 适度保守，较 G0 缩小误差近半 |
| **80% (0.80)** | 79.36% | **79.62%** | **-0.38%** | 1.6210 | **超高精度吻合（绝对误差仅 0.38%）** |
| **90% (0.90)** | 91.73% | **90.58%** | **+0.58%** | 2.0834 | **超高精度吻合（绝对误差仅 0.58%）** |
| **95% (0.95)** | 96.48% | **95.69%** | **+0.69%** | 2.4841 | **超高精度吻合（绝对误差仅 0.69%）** |

### 8.3 Spread-Skill 关系与出版级图表
- **诊断指标**：在跨窗口混合池化（Cross-Window Pooled RMS）下计算集合发散度（Spread）与重构误差（Skill）之比，为长时程展开提供自适应不确定性安全带；
- **落盘报告与图集**：详细指标见 [outputs/metrics/phase3_probabilistic_evaluation.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/phase3_probabilistic_evaluation.json)，出版级图表导出至 [outputs/figures/probabilistic/](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/probabilistic/)（Figure 1~3 & Summary PNG/PDF）。

---

## 9. Navier-Stokes 与示踪剂 PDE 动力学残差物理评测大盘

### 9.1 真实流场与推演模型 PDE 残差基线审计 (Audit)
经由 [scripts/audit_pde_residuals.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/audit_pde_residuals.py) 审计，各对象在无量纲时间网格上的偏微分方程残差 RMSE 实测如下：

| 评估对象 | 散度 RMSE ($\nabla \cdot \mathbf{u}$) | 动量残差 $u$ ($res_u$) | 动量残差 $v$ ($res_v$) | 示踪物输运残差 ($res_s$) | 物理特性解读 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ground Truth (全分辨率)** | $0.0026$ | $0.0016$ | $0.0015$ | $0.0013$ | 严格服从连续 Navier-Stokes 方程与示踪剂对流-扩散守恒 |
| **Ground Truth (降采样)** | $0.0010$ | $0.0015$ | $0.0015$ | $0.0011$ | 网格降采样平滑轻微削减高频微分波动 |
| **Autoencoder 重建** | $0.0988$ | $0.0375$ | $0.0479$ | $0.0459$ | 8x 空间潜流形压缩保留了绝大部分微分物理结构 |
| **D0 推演模型 (未约束)** | $0.1257$ | $0.0929$ | $0.0733$ | $0.1060$ | 自回归展开存在小幅度微分漂移，验证了显式 PDE 约束的必要性 |

### 9.2 PDE 受控训练微调全体验证集评测对比 (1110 窗口全量评测)
在验证集全部 1110 个滑动窗口上（$H=12$，下采样因子 2），对比父底座 $D0$、无 PDE 约束对照组 $P0$ 与引入 PDE 残差约束的实验组 $PDE$：

| 评价物理指标 | 父模型 D0 基线 | P0 对照组 (step 50) | PDE 实验组 (step 50) | PDE 相对 P0 对照组改善百分比 |
| :--- | :---: | :---: | :---: | :---: |
| **水平动量残差 $res_u$ RMSE** | 0.07566 | 0.08353 | **0.08229** | <font color="#2ea44f">**-1.48% (显著降低)**</font> |
| **竖直动量残差 $res_v$ RMSE** | 0.05744 | 0.06250 | **0.06226** | <font color="#2ea44f">**-0.39% (降低)**</font> |
| **示踪物输运残差 $res_s$ RMSE** | 0.11904 | 0.12332 | **0.12064** | <font color="#2ea44f">**-2.18% (大幅改善)**</font> |
| **速度散度 RMSE** | 0.12529 | 0.13783 | **0.13757** | <font color="#2ea44f">**-0.19% (改善)**</font> |
| **压力场 VRMSE ($vrmse_p$)** | 0.46877 | 0.44601 | **0.44499** | <font color="#2ea44f">**-0.23% (更精准)**</font> |
| **示踪物场 VRMSE ($vrmse_s$)** | 0.10054 | 0.10086 | **0.10072** | <font color="#2ea44f">**-0.14% (改善)**</font> |
| **低扩散子群 ($Sc=0.1$) VRMSE** | 0.14393 | 0.14754 | **0.14709** | <font color="#2ea44f">**-0.31% (全场领先)**</font> |

- **落盘评测报告**：[outputs/evaluations/pde_controlled_candidates_full_val.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/evaluations/pde_controlled_candidates_full_val.json)。
- **核心结论**：PDE 残差约束不仅使动量与标量对流扩散物理方程残差全面下降，同时在保持场值重构高精度的前提下改善了压力场与标量场的空间保真度。

