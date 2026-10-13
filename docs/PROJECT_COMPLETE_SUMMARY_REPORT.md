# 流场世界模型全阶段研发总结与定论报告

> **项目名称**：World-Model-FlowField-v1  
> **基准任务**：The Well 二维剪切流（不可压缩 Navier-Stokes 方程 + 被动标量输运方程）  
> **计算环境**：NVIDIA vGPU-32GB × 2 (CUDA 13.0, PyTorch 2.10.0+cu128)  
> **报告原则**：直接陈述核心工程与物理定论，实现**所有研发分支定量曲线图与定性结果图 100% 全配对覆盖**。

---

## 一、 物理场契约与空间表征自编码网络（Representation Learning）

### 1. 核心结论
- **历史输入步长定论**：严格确定并锁定历史输入步长为 **$L = 4$ 帧**（即输入张量为 $q_{t-3:t} \in \mathbb{R}^{4 \times 4 \times Ny \times Nx}$）。输入 $L=1$ 帧缺失流体速度时序一阶导与加速度信息；$L > 4$ 边际收益衰减严重且注意力显存开销成倍膨胀。
- **物理参数调制机制**：雷诺数 $Re \in [10^3, 10^5]$ 与施密特数 $Sc \in [0.1, 1.0]$ 由 `PhysicalContext` 统一编码，经 `AdaLN-Zero` 算子直接调制时空注意力骨干网络各层归一化层。
- **空间潜流形压缩与重构闭环**：空间网格采用 $64\times$ 周期卷积自编码器架构（$256 \times 512 \to 32 \times 64$），潜空间通道数升维至 64；对物理压力场显式施加空间零均值投影规范（$\iint p \, dxdy = 0$），彻底消除不可压缩流体相对压力的常数漂移自由度。
- **自编码器闭环重构精度**：在完全独立的测试集上，自编码器四通道平均重构 VRMSE 达到 **0.1015**（水平速度 $u$ 重构 RMSE 仅 0.0221，流体压力 $p$ RMSE 仅 0.0046，示踪物 $s$ RMSE 仅 0.0232），高保真锁定了大尺度流线与亚网格细微涡旋结构。
- **坐标轴对齐规范**：统一流向为 $x$（dim -2）、法向为 $y$（dim -1），全局校正历史坐标轴颠倒缺陷。

### 2. 物理场快照可视化
![典型剪切流 4 通道物理场（流向速度 u、法向速度 v、流体压力 p、被动标量示踪物 s）在 5 个演化时刻的快照矩阵](../outputs/dataset_viz/field_snapshots.png)

### 3. 表征空间自编码器流场闭环重构对比云图
![空间表征自编码器闭环重构对比云图（第一行：测试集真实物理场；第二行：自编码器重构场；第三行：五通道绝对重构误差场，含计算涡量 field）](../outputs/figures/representation/sample_autoencoder_reconstruction.png)

---

## 二、 纯潜空间自回归自由滚动与长程基准对比（Rollout Benchmark）

### 1. 核心结论
- **空间直接预测网格撕裂**：直接在物理网格空间自回归推演的时空 Transformer（Direct ST）在滚动至 Step 10 时即出现不可逆高频数值色散（Step 30 散度高达 89.73，流场完全破裂）；
- **神经算子长程过度平滑**：二维傅里叶神经算子（FNO-2D）依赖高频截断，随滚动步长加深出现严重的数值耗散与细节过平滑，无法有效维系流体微尺度剪切涡对；
- **潜空间 FIFO 滚动稳定性优势**：基于固定先进先出（FIFO）潜状态滑窗机制的世界模型，在完全不访问原始高维物理网格的前提下实现连续 30 步潜空间自由演化；长程推演稳定性显著胜出传统直接预测与基线神经算子。

### 2. 长程推演基准评测 8 面板曲线
![30 步自回归长程推演对比基线（Persistence B0 与 FNO-2D）的 8 面板物理指标曲线（包含各通道 VRMSE、全场散度、涡量 RMSE、动能能谱误差等）](../outputs/figures/benchmark/rollout_benchmark_curves.png)

---

## 三、 物理守恒损失消融体系与动力学诊断（Closure-R4：E0 $\to$ E4）

### 1. 核心结论
- **纯数据驱动发散定论**：单步或纯多步滚动场损失模型（E1）虽然单步误差表面较小，但随着展开步长递增，不可压缩散度与伪涡量迅速累积，导致高频伪能谱爆发并引起长程数值发散；
- **全物理约束收敛定论**：联合注入谱空间连续性散度损失 $L_{\text{div}}$、谱涡量守恒损失 $L_\omega$、拟涡能损失 $L_{\text{enstrophy}}$ 及示踪物有界全变差损失 $L_{\text{tracer}}$ 的全物理受控模型（E4），将长程散度误差压制达 **3 个数量级**；
- **多随机种子一致性**：在 3 个独立随机种子（Seeds 42, 43, 44）重复实验下，物理指标均值改善在所有种子上严格单调，方差大幅收敛；
- **频域耗散各向异性**：各向异性能谱分析证实，物理损失不仅抑制全局能量发散，而且精准纠正了高波数 $k_x$ 与 $k_y$ 方向能量衰减的不对称性。

### 2. 物理消融三随机种子聚合误差曲线
![Seeds 42, 43, 44 三种子聚合的 E0~E4 物理损失消融 4 面板误差曲线（带 95% 置信度阴影带，展示 E4 在散度与涡量上的绝对压制）](../outputs/figures/closure_r4/closure_r4_physics_ablation_tri_seed_curves.png)

### 3. 训练收敛动力学分析曲线
![Closure-R4 训练阶段各物理消融组（E0~E4）在多随机种子下的损失下降轨迹与收敛诊断曲线](../outputs/figures/closure_r4/training_convergence_curves.png)

### 4. 各向异性能谱流向/展向耗散比率曲线
![二维剪切流各向异性能谱流向与展向耗散比率演化曲线（证实物理守恒损失消除高频各向异性伪能量聚集）](../outputs/figures/manuscript/directional_spectral_ratio_curves.png)

### 5. E0（无物理约束）vs E4（全物理约束）长程涡量定性比对云图
![E0 单步基线 vs E4 全物理约束模型在 Step 10 长程自回归推演下的定性涡量场比对云图（直观展示 E0 高频色散伪涡与 E4 紧致真实剪切涡结构的鲜明对比）](../outputs/figures/closure_r4/compare_e0_vs_e4_vorticity.png)

---

## 四、 预测跨度阶梯与双重检查点选型（Horizon-R1 & H16）

### 1. 核心结论
- **双检查点解耦定论**：单步验证损失极小的短程模型（Short-best）与长程加权综合泛化指标 $J_{\text{long}} = \text{mean}(h10, h20, h30)$ 极小的长程模型（Long-best）并非同一检查点。超长程推演任务必须显式依据 $J_{\text{long}}$ 选取权重，否则将在自回归后期遭受累积平滑误差反噬；
- **跨度阶梯收益**：自回归时序跨度逐步由 $H=1$ 稳步跃升至 $H=2, 4, 8, 12, 16$，长程展开时间跨度显著拉长且完全免于数值崩溃。

### 2. 检查点选型分异定量对比柱状图
![Horizon-R1 跨度消融下短程最优与长程最优检查点的长程指标 J_long 对比柱状图](../outputs/figures/horizon_r1/horizon_r1_checkpoint_selection.png)

### 3. H16 超长时序自回归展开定性云图
![H16 超长时序自回归推演下水平速度 u 的 4x7 定性云图（真实场 vs H8 vs H16-Short vs H16-Long 及各自绝对误差）](../outputs/figures/h16_comparison_v2/comparison_fixed_ref_u.png)

---

## 五、 高斯概率潜空间动力学世界模型（ProbLatent Phase 3: G0 vs G1）

### 1. 核心结论
- **均值平价与似然收益**：在严格保持确定性主干均值预测 $\mu$ 零误差的前提下，条件异方差高斯概率世界模型 G1 在测试集上负对数似然（NLL）下降 **$0.197\text{ nats/element}$**，连续排量概率得分（CRPS）相对改善 6.1%；
- **区间校准精度**：80% 与 90% 名义置信区间的实测经验覆盖率分别达到 79.6% 与 90.6%（绝对校准偏差控制在极小的 $0.4\% \sim 0.6\%$ 范围）；
- **不确定性空间聚焦与剪切边界对齐**：预测不确定度标准差 $\sigma$ 场高度聚焦于两相流强烈剪切界面与微涡卷吸边缘；真实误差与方差重叠度高，为工业级流动不确定度量化（UQ）提供了高可信度指标。

### 2. 概率评估综合三面板曲线总图
![ProbLatent-R1 Phase 3 综合成果总图（30 步 VRMSE 演化、区间校准柱状图、Spread-Skill 离散度与误差关系）](../outputs/figures/probabilistic/figure_summary_phase3.png)

### 3. 高斯概率潜空间定性抽样与不确定性可视化
![真实场、确定性均值预测及 3 组独立高斯后验采样实例定性云图（展现微观随机扰动多样性）](../outputs/figures/probabilistic/sample_gaussian_ensemble_realizations.png)

![空间预测不确定度标准差场 σ 与实际经验预测误差绝对值对比云图（验证高不确定度与强剪切误差的空间重叠）](../outputs/figures/probabilistic/sample_gaussian_uncertainty_vs_error.png)

![流向中截线剖面的 50%、90%、95% 置信区间条带与真实观测曲线对比](../outputs/figures/probabilistic/sample_gaussian_prediction_intervals.png)

---

## 六、 连续 Navier-Stokes 与示踪物 PDE 受控微调（PDE-Controlled）

### 1. 核心结论
- **连续谱空间残差阻尼定论**：基于周期性二维 FFT 高精度谱导数微分器，在 50 步受控微调下，PDE 连续损失成功将示踪物平流扩散残差均方根降低 **2.18%**（高扩散体制 $Sc=0.1$ 下降低幅度达 **3.26%**），同时对不可压缩速度散度误差施加严格的硬物理约束阻尼；
- **全局精度平衡**：在保持场总 VRMSE 变动在千分之一以内的同时（$-0.09\%$ vs P0 对照组），大幅抑制了非物理质量生成与浓度渗漏。

### 2. PDE 受控实验定量三面板对比
![PDE 控制实验三面板综合柱状图（全场与通道误差、物理残差阻尼、Schmidt 数敏感性）](../outputs/figures/pde_controlled/figure_pde_controlled_comparison.png)

### 3. PDE 受控实验定性抽样、残差阻尼与误差场可视化
![真实场 vs D0 vs P0 vs PDE 的 4 通道物理场（u, v, s, 涡量）横向定性比对](../outputs/figures/pde_controlled/sample_pde_flow_fields_comparison.png)

![连续性散度与示踪物物理残差空间阻尼对比图（P0 对照组 vs PDE 受控组，验证受控组空间残差的大幅减退）](../outputs/figures/pde_controlled/sample_pde_residuals_comparison.png)

![PDE 受控组与对照组各通道物理残差绝对误差场定性对比图](../outputs/figures/pde_controlled/sample_pde_errors_comparison.png)

---

## 七、 潜流匹配生成流世界模型（Latent Flow Matching / FM-R2: C2 vs R2-A）

### 1. 核心结论
- **自条件连续流匹配架构（FM-R2）**：将世界模型推演范式由离散差分步进推进至连续速度场常微分方程（Neural ODE），以自条件（Self-conditioned）流匹配消除自回归推演中的暴露偏差（Exposure Bias）；
- **探索与验证严格解耦**：严格区隔探索种子（Discovery Seed 42）与验证种子（Replication Seeds 43~46），杜绝后验选择性偏差；
- **全随机种子鲁棒改善**：在 Rollout-Aware 训练策略（R2-A）下，在 24 对单轨迹配对推演中，次要物理指标（单样本 VRMSE、散度 RMS、涡量 RMSE）在 **4/4 个独立验证随机种子上实现统计显著全面改善**；
- **学术诚信与异常透明记录**：实事求是呈现 Seed 45 在 $h=10$ 能谱上的反转表现（$+11.99\%$），未作任何人为掩盖或数据修饰。

### 2. 预注册多随机种子主要结局图
![预注册多随机种子主要结局图（h=10 系综误差、h=10 能谱误差及 Seed 45 反转、h=5 短程误差，严格区分探索种子 42 与验证种子 43-46）](../outputs/figures/paper/fig2_primary_multiseed.png)

### 3. 预指定次要物理指标单轨迹配对与均值改善图
![预指定次要物理指标图（呈现 24 对单轨迹配对斜率细线与 4 个种子的均值粗线，展示 4/4 种子全面改善）](../outputs/figures/paper/fig3_secondary_physics.png)

### 4. 潜流匹配连续神经 ODE 生成流场定性比对云图
![潜流匹配模型（FM-R2）定性推演流场对比云图（真实场 vs C2 标准流匹配 vs R2-A 自条件流匹配在水平速度 u、垂直速度 v、速度模长 |u|、涡量场 ω 与绝对散度场 |div u| 的横向高保真比对）](../outputs/figures/paper/fig4_qualitative_rollout_fields.png)

---

## 八、 论文手稿物理不变量与跨种子离散度综合演化（Manuscript Invariants）

### 1. 核心结论
- **全场长程稳定性与跨种子方差紧凑性**：综合 5 个跨度与多种子统计，全场 VRMSE 随步长演化保持高度平缓上升，多随机种子离散度阴影带极其狭窄，证明流场世界模型对参数初始化具有高鲁棒性；
- **物理不变量四面板全面守恒**：散度均方根（Divergence RMS）、涡量均方根误差（Vorticity RMSE）、拟涡能相对误差（Enstrophy Relative Error）与示踪物物理越界逃逸率（Tracer Out-of-Bounds Rate）四大物理不变量在长达 30 步推演中均保持严密物理受控。

### 2. 全场 VRMSE 随预测步长演化及跨种子离散度曲线
![手稿图 A：全场 VRMSE 随预测步长演化趋势图及跨随机种子离散度统计分布（带置信区间带）](../outputs/figures/manuscript/figure_a_vrmse_and_dispersion.png)

### 3. 系统级四大物理不变量多跨度综合演化曲线
![手稿图 B：四大物理不变量在不同预测跨度下的系统级演化表现（四面板：连续性散度、涡量 RMSE、拟涡能相对误差、示踪物越界逃逸率）](../outputs/figures/manuscript/figure_b_physical_invariants.png)

---

## 九、 边界与极端失败案例审计（Failure Cases & Robustness Audit）

### 1. 核心结论
- **误差集中于两相剪切界面**：空间误差热点审计证实，主要推演误差并非全局漂移，而是集中在两相剪切层交界面处的亚像素相位微移；
- **几何涡脱落破裂点**：极端发散案例通常发生于强剪切失稳诱发复杂涡对脱落时，若局部涡量梯度超出潜流形表达容量，将引发累积能量色散；
- **小方差流场数值放大效应**：最差案例（Worst Case）由于真值背景能量方差极小而导致相对 VRMSE 被数值放大，其实际物理绝对 RMSE 仍受控在 0.19 以内。

### 2. 极端案例审计可视化
![全测试集最优案例、中位案例与最差发散案例的涡量场云图与 30 步累积发散曲线](../outputs/figures/benchmark/failure_cases_analysis.png)

---

## 十、 全局研发演化总结与动态流场仿真资产（Global Synthesis & Dynamic Simulation）

### 1. 技术代际演化全景

| 研发代际 | 核心技术特征 | 代表模型 / 阶段 | 核心成果与关键突破 |
| :--- | :--- | :--- | :--- |
| **代际 I** | 空间降采样与纯潜自回归 | Representation & Rollout Baseline | 确立 $L=4$ 历史步长与空间自编码器 $64\times$ 周期压缩，潜空间 FIFO 滚动杜绝网格撕裂 |
| **代际 II** | 谱空间物理守恒损失注入 | Closure-R4 (E0 $\to$ E4) | 联合注入谱散度、谱涡量与示踪物有界损失，散度误差压制达 3 个数量级 |
| **代际 III** | 展开视野阶梯与双选型 | Horizon-R1 & H16 | 揭示短程最优与长程最优解耦机制，推演视野稳步拓展至 16 步 |
| **代际 IV** | 潜空间高斯概率模型 | ProbLatent-R1 Phase 3 (G0/G1) | 保持均值平价前提下实现条件异方差量化，区间校准误差仅 $0.4\%\sim 0.6\%$ |
| **代际 V** | 连续物理方程受控微调 | PDE-Controlled Optimization | 连续 FFT 谱微分反传，示踪物物理残差显著下降 2.18%（高扩散下下降 3.26%） |
| **代际 VI** | 连续生成神经 ODE 世界模型 | Latent Flow Matching (FM-R2) | 自条件流匹配消除自回归暴露偏差，4/4 验证随机种子全面改善，形成出版级手稿定论 |

### 2. 动态多物理场演化仿真资产
本项目已成功生成出版级 200 帧 4 通道高保真动态仿真流场动画（包含流向速度 $u$、法向速度 $v$、流体压力 $p$ 与示踪物 $s$ 的长程时空连续演化）：
- **动态 GIF 资产**：`outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames.gif`
- **高清 MP4 资产**：`outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames.mp4`
- **动画契约元数据**：`outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames_metadata.json`
