# 流场世界模型 V1 研发路线图与工程 TodoList

> **项目名称**：World-Model-FlowField-v1  
> **基准数据集**：The Well `shear_flow` (2D 不可压缩剪切流 + 被动示踪�> **项目名称**：World-Model-FlowField-v1  
> **基准数据集**：The Well `shear_flow` (2D 不可压缩剪切流 + 被动示踪标量)  
> **计算环境**：NVIDIA vGPU-32GB × 2 (CUDA 13.0, PyTorch 2.10.0+cu128)  
> **工程测试基线**：全套自动化测试套件通过（45/45 tests passed，含协议契约测试）

---

## 一、 整体研发路线图总览看板

| 阶段 | 核心任务 | 关键目标 / 验证标准 | 状态 |
| :--- | :--- | :--- | :--- |
| **Stage 1** | 数据基座与物理规范 | HDF5 数据审计、时间滑动窗口、物理参数归一化、测试集防泄漏协议（Sc 留出就绪，Re 留出因本地单参数标记为 BLOCKED_BY_DATA） | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 2** | 空间潜状态编码器与解码器 | 8x 下采样卷积结构、双向周期边界、压力零均值非就地投影（评估阶段）、四场重建闭环 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 3** | 物理条件层与因子化 Transformer | $Re, Sc \to \text{log} \to \text{MLP} \to \text{AdaLN}$、时空解耦注意力、潜状态残差更新 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **基线体系** | 统一基线与公平评测竞技场 | Persistence、FNO-2D、Direct ST Transformer 接口统一 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 4.1** | 潜空间自由滚动机制 | `HistoryBuffer` 纯潜空间自回归推演、30 步滚动评测矩阵构建 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 4.2** | 双卡 DDP 短程自由滚动长训练 | 引入多步滚动监督 ($H=2$) 抑制自回归自激发散、双卡分布式训练 30 Epochs | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 4.3** | 自由滚动收敛复评 | 载入单步底座与滚动长训双模型展开 30 步评测，确认单步误差暴降 80.5% | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 5** | 物理守恒损失消融实验 (E0 - E4) | 实验流水线就绪（旧四组实验为 Historical；Closure-R2 E0-E4 全量重跑待调度） | <font color="#d29922">● 流水线 PASS / 跑批 PENDING</font> |
| **Stage 6** | 周期 FFT 导数与独立物理评价系统 | 二维周期谱导数与拉普拉斯算子、全量物理衍生量、8 联排出版级 Rollout 看板导出 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **V1 全链验收** | 汇总主模型与基线、失败案例审计、架构规范与 V1 验收 | 协议契约全面冻结（P1-1~P1-7 闭环），待执行最终 E0-E4 与 H=4/8 统一超参重跑 | <font color="#d29922">● 进行中 (IN PROGRESS)</font> |
| **10 月路线图** | 课程式长推演、潜流形扩散、宽域泛化与三维预研 | 50~100 步课程式自回归、Latent Diffusion 湍流随机分岔、Re 宽域外推与 3D 周期谱导数 | <font color="#8c959f">○ 待进行 (PENDING)</font> |30 步评测，确认单步误差暴降 80.5% | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 5** | 物理守恒损失消融实验 ($L_{\text{field}}$, $+L_{\text{div}}$, $+L_\omega$, $+L_{\text{div}}+L_\omega$) | 四组消融双卡并发训练、30 步多尺度推演、单步与长期滚动物理守恒对比 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 6** | 周期 FFT 导数与独立物理评价系统 | 二维周期谱导数与拉普拉斯算子、全量物理衍生量、8 联排出版级 Rollout 看板导出 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **V1 全链验收** | 汇总主模型与基线、失败案例审计、架构规范与 V1 验收 | 全模型横向竞技大盘、极端边界误差归因、系统架构更新、形成 V1 阶段验收通过定论 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **10 月路线图** | 课程式长推演、潜流形扩散、宽域泛化与三维预研 | 50~100 步课程式自回归、Latent Diffusion 湍流随机分岔、Re 宽域外推与 3D 周期谱导数 | <font color="#8c959f">○ 待进行 (PENDING)</font> |

---

## 二、 各阶段详细任务分解与执行证据

### Stage 1: 数据基座与物理规范
- [x] **数据格式实测审计**：解析 HDF5 层级群组，确认标量场 `t0_fields`（`pressure`, `tracer`）与向量场 `t1_fields`（`velocity`）键名与轴序。
  - *产出文档*：[docs/DATA_AUDIT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/DATA_AUDIT.md)
- [x] **物理参数归一化**：实现 $\tilde{c} = [\log_{10}(Re), \log_{10}(Sc)]$ 对数映射，稳定多量级参数梯度。
- [x] **防数据泄漏协议**：识别官方划分中跨 $Sc$ 轨迹流场重叠缺陷，制定独立轨迹划分隔离方案。
- [x] **数据加载流水线**：实现支持滑动窗口 $L=4$ 与 $H \ge 1$ 滚动采样的 `ShearFlowDataset` 与 `get_dataloader`。
  - *代码位置*：[src/data/shear_flow_dataset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/shear_flow_dataset.py)

---

### Stage 2: 空间潜状态编码器与解码器
- [x] **8x 空间下采样/上采样结构**：构建 `Encoder2D` 与 `Decoder2D`，将 $128 \times 128 \times 4$ 压缩为 $16 \times 16 \times 64$ 潜空间，压缩率达 64 倍。
- [x] **周期边界保证**：网络卷积层全面采用 `padding_mode="circular"`，严格吻合物理剪切流双向连续性。
- [x] **压力零均值非就地规范投影**：
  - 数学公式：$p \leftarrow p - \frac{1}{|\Omega|}\sum p$
  - 实现采用非就地（out-of-place）张量拼接，兼容 PyTorch 反向传播与评估模式切换；
  - 实测残余均值绝对值 $< 2.5 \times 10^{-7}$，满足数值零均值规范。
- [x] **四场重建闭环评价与权重保存**：
  - 最佳检查点：[outputs/checkpoints/representation/best_vrmse_mean.pt](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/checkpoints/representation/best_vrmse_mean.pt)
  - 指标落盘：[outputs/metrics/representation_metrics.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/representation_metrics.json)

---

### Stage 3: 动力学条件层与因子化 Transformer
- [x] **物理条件注入层**：
  - 构建 `ConditioningMLP` 与 `AdaptiveLayerNorm2D` (AdaLN)；
  - 实施 **Zero-Init 初始化**（调制因子的线性头全零初始化），确保模型在初始步等价于标准 LayerNorm，提升训练初期平稳性。
  - *代码位置*：[src/models/conditioning.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/conditioning.py)
- [x] **因子化时空解耦 Transformer (`LatentSTTransformer`)**：
  - 空间自注意力（捕获全场大尺度涡旋拓扑结构）与时间自注意力（捕捉对流输运时序相关性）交替堆叠；
  - 支持 `residual` 模式（$\hat{Z}_{t+1} = Z_t + \Delta Z$）与 `direct` 直接预测模式。
  - *代码位置*：[src/models/latent_transformer.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/latent_transformer.py)

---

### 基线体系: 统一标准与公平评测竞技场
- [x] **PersistenceBaseline**：物理惯性保持基准（输出最近一帧输入）。
- [x] **FNO2D (Fourier Neural Operator)**：复数谱域截断卷积网络，支持任意物理连续网格。
- [x] **PDETransformer (Direct ST Transformer)**：在物理原始网格切 patch 进行时空注意力推演的代表性直接预测基线。
- [x] **统一接口契约**：三大基线与世界模型统一在 [src/baselines/__init__.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/baselines/__init__.py) 导出，且 [scripts/train_forecaster.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_forecaster.py) 原生支持 `--model` 一键切换。

---

### Stage 4: 潜空间自由滚动与多步自回归
- [x] **自回归滑动缓冲区 (`HistoryBuffer`)**：
  - 严格潜空间闭环：历史帧编码为潜状态后，所有中间推演仅在潜空间内进行；
  - 坚决杜绝读取未来真实帧（No Data Leakage）；
  - 滚动结束后一次性解码回物理场，大幅节省显存与计算开销。
  - *代码位置*：[src/models/history_buffer.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/history_buffer.py)
- [x] **30 步滚动基准评测矩阵**：
  - 评测脚本：[scripts/evaluate_rollout.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_rollout.py)
  - 评测结果已落盘：[outputs/metrics/rollout_benchmark.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/rollout_benchmark.json)
  - *核心科学发现*：PDE-Transformer（网格直接切片）在单步 VRMSE 虽达 0.8511，但在展开至 30 步时发生剧烈数值色散（VRMSE 飙升至 26.83，散度达 89.73），证实了潜空间表示对微分守恒与长期稳定性的决定性优势。
- [x] **双卡 DDP 短程自由滚动长训练任务**：
  - 启动命令：`torchrun --nproc_per_node=2 scripts/train_forecaster.py --model latent_transformer --output_dir outputs/checkpoints/dynamics/stage4_latent_rollout_ddp --horizon 2 --epochs 30 --batch_size 4 --use_amp`
  - 训练结果：**30 个 Epochs 全部收敛完成**。Train Loss 降至 **$2.3938 \times 10^{-5}$**，Val VRMSE 达到最佳 **1.1526**（$u$ 速度场 2.74%，$s$ 示踪标量 5.16%）；
  - 最佳权重已落盘：[outputs/checkpoints/dynamics/stage4_latent_rollout_ddp/latent_transformer/best_vrmse_mean.pt](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/checkpoints/dynamics/stage4_latent_rollout_ddp/latent_transformer/best_vrmse_mean.pt)。
- [x] **自由滚动收敛复评**：
  - 在统一测试集（36 条独立轨迹）上对四大模型及潜空间新旧权重展开 30 步全量对比；
  - 评测指标已落盘：[outputs/metrics/rollout_benchmark.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/rollout_benchmark.json) 与 [docs/BENCHMARK_RESULTS.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/BENCHMARK_RESULTS.md)；
  - **核心验证**：短程滚动监督使单步预测误差由 1.7842 暴降至 **0.3481**（降幅达 **80.5%**），单步涡量误差降低 **57.8%**，夺得全场学习模型单步预测冠军。

---

### Stage 5: 物理守恒损失消融实验 ($L_{\text{field}}$, $+L_{\text{div}}$, $+L_\omega$, $+L_{\text{div}}+L_\omega$)
依据项目规范第 9.5 节与 9/29 研发目标，完成物理约束系统消融实验与深度量化：
- [x] **四组消融模型双卡并发训练**：
  - 调度脚本：[scripts/run_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_physics_ablation.py)
  - 训练日志与最佳检查点已全量落盘（耗时共约 100 分钟，双卡满载并发加速）；
- [x] **Group 1 ($L_{\text{field}}$ 基线)**：纯数据驱动重构，$\lambda_{\text{div}}=0, \lambda_\omega=0$；
- [x] **Group 2 ($+L_{\text{div}}$ 无散度约束)**：$\lambda_{\text{div}}=0.01$，**单步散度 RMSE 暴降 84.5%**（从 4.564 降至 0.708）；
- [x] **Group 3 ($+L_\omega$ 涡量拓扑守恒)**：$\lambda_\omega=0.05$，**单步涡量误差降幅 43.4%**（0.2896 全场最低），**彻底压制拟能（Enstrophy）虚拟爆炸**（降幅超 99.4%）；
- [x] **Group 4 ($+L_{\text{div}}+L_\omega$ 全物理约束)**：$\lambda_{\text{div}}=0.01, \lambda_\omega=0.05$，**长期自由滚动之王**：
  - 彻底摆脱基线的累积发散，构建了超稳定的“守恒平台”；
  - **Step 10 VRMSE 较基线暴降 92.0%**（从 39.28 压制至 3.12）；
  - **Step 30 真实场 RMSE 为 0.1774（全场最优，较基线降低 46.2%）**；
- [x] **30 步多模型自由滚动评测与指标落盘**：
  - 评测脚本：[scripts/evaluate_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_physics_ablation.py)
  - 指标数据落盘：[outputs/metrics/physics_ablation_benchmark.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/physics_ablation_benchmark.json)
- [x] **出版级消融对比曲线看板生成**：
  - 绘图脚本：[scripts/plot_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/plot_physics_ablation.py)
  - 高清对比大图导出：[outputs/figures/physics_ablation_curves.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/physics_ablation_curves.png)
- [x] **理论与物理机理总结固化**：详见 [docs/BENCHMARK_RESULTS.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/BENCHMARK_RESULTS.md) 第 5 章。

---

### Stage 6: 周期快速傅里叶变换导数与全部独立物理评价指标
- [x] **周期快速傅里叶变换导数与微分算子库**：
  - 二维周期谱梯度 `spectral_grad_2d`、散度 `compute_divergence`、涡量 `compute_vorticity`；
  - 高精度 2D 周期拉普拉斯算子 `compute_laplacian_2d`（解析解相对误差达机器极限 $1.40 \times 10^{-12}$）；
  - 平均动能密度 `compute_kinetic_energy` 与拟能密度 `compute_enstrophy`；
  - 零均值压力规范投影 `project_zero_mean_pressure`；
  - *代码位置*：[src/utils/fft_derivatives.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/utils/fft_derivatives.py)
- [x] **全物理独立评价系统**：
  - 完整覆盖场误差（VRMSE, RMSE, MSE, Max Error）、散度（RMSE & Max）、涡量 RMSE、动能相对误差、拟能相对误差、能谱 MAE 及低中高频段分解、Tracer 方差保持率/质量误差/均值误差/越界率；
  - *代码位置*：[src/metrics/rollout.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/metrics/rollout.py), [src/metrics/field.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/metrics/field.py), [src/metrics/tracer.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/metrics/tracer.py)
- [x] **出版级 8 联排 Rollout 物理曲线看板生成器**：
  - [scripts/plot_rollout_comparison.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/plot_rollout_comparison.py)
  - 成功导出高清（300 DPI）全景对比图：[outputs/figures/rollout_benchmark_curves.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/rollout_benchmark_curves.png)
- [x] **测试套件全量通过**：
  - `pytest` 39 项测试全部 100% 绿灯 PASS。

---

### V1 阶段全链验收与失败案例审计 (9/30)
- [x] **主模型与全量基线横向竞技大盘汇总**：
  - 汇总 Persistence、FNO-2D、PDE-Transformer、单步潜模型、短程滚动潜模型与全物理消融最优模型在 Step 1/10/30 的全套指标；
  - 验证了潜空间双物理约束模型在 Step 30 达到全场最低绝对场误差（RMSE=0.1774）；
- [x] **36 条测试轨迹逐条误差审计与失败案例分析**：
  - 审计脚本：[scripts/analyze_failure_cases.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_failure_cases.py)
  - 导出典型对比图：[outputs/figures/failure_cases_analysis.png](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/failure_cases_analysis.png)
  - 定位退化根因：小扰动场方差归一化放大效应、高剪切交界面 1~2 像素相位微移、低 Schmidt 数被动标量数值扩散；
- [x] **系统架构说明与技术规范文档编写**：
  - 产出规范：[docs/ARCHITECTURE.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/ARCHITECTURE.md)
  - 全面图解与公式化定义：潜流形编解码、AdaLN-Zero 调制、HistoryBuffer 循环、FFT 谱导数梯度穿透链；
- [x] **V1 阶段全链验收报告固化**：
  - 验收报告：[docs/V1_ACCEPTANCE_REPORT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/V1_ACCEPTANCE_REPORT.md)
  - 依据代码规范与测试证据判定：**审查结论 PASS**。

---

### 10 月份下一轮研发任务路线图 (October Roadmap)
- [ ] **任务 1：课程式长时程多步自回归展开 ($H=4, 8$)**
  - 设计按 Epoch 递进展开步长的训练器；
  - 引入 Gradient Checkpointing 机制，压制长程计算图显存峰值；
  - 目标：攻克 50~100 步超长期推演无能量衰减。
- [ ] **任务 2：潜流形生成式扩散世界模型 (Latent Diffusion Flow Model)**
  - 在空间潜序列接入条件扩散模型（DiT），建模高 Reynolds 数湍流破裂的多模态随机分岔；
  - 目标：解决传统 MSE 回归在高频旋涡细丝上的过度平滑（模糊）缺陷。
- [ ] **任务 3：宽参数域泛化与极端工况外推适应性**
  - 评估并提升对 $Re = 5 \times 10^5$、$Sc = 5.0, 10.0$ 的外推鲁棒性；
  - 构建基于物理动量边界层厚度的特征嵌入层。
- [ ] **任务 4：三维不可压缩流场与复杂几何架构预研**
  - 3D 周期谱导数算子与 3D 涡矢量计算库实现；
  - 分布式张量并行潜空间显存负载评估。

---

## 三、 硬件能效说明：显存占用与功率利用率

在实际多卡训练中，出现**显存占用约为 9GB / 32GB（未满），但 GPU 功率与计算利用率达到 90%~100%（满载）**，这是高能效架构设计的典型特征：

1. **显存未占满的原因**：
   - 编码器将物理网格分辨率从 $128 \times 128$ 下采样 8 倍压缩为 $16 \times 16$ 潜空间，特征点数缩减了 64 倍；
   - 极大降低了自回归展开过程中的激活值缓存需求，彻底规避了爆显存（OOM）风险。
2. **功率与利用率打满的原因**：
   - Transformer 的注意力计算密集（包含多层多头自注意力 GEMM 矩阵乘、Softmax、MLP 门控及 AdaLN 条件调制）；
   - 双卡 DDP 并行与 PyTorch AMP 混合精度调度极度饱满，数据传输零 I/O 阻塞；
   - GPU 的 Tensor Core 和 CUDA Core 始终处于高频浮点吞吐状态，处于最优的 **Compute-Bound（计算受限）** 高能效工况。
