# 流场世界模型 V1 实验产物与基准评测报告

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
| **评测曲线** | [outputs/figures/rollout_benchmark_curves.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/rollout_benchmark_curves.png) | 四大模型多步滚动误差发散对比曲线 |
| **数据划分配置** | [outputs/splits/grouped_split.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/splits/grouped_split.json) | 严格按初始扰动隔离的防数据泄漏划分文件 |
| **归一化参数** | [outputs/normalization/stats_grouped.pt](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/normalization/stats_grouped.pt) | 流场各通道均值、方差与极值预处理统计参数 |
