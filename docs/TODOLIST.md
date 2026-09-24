# 流场世界模型 V1 研发路线图与工程 TodoList

> **项目名称**：World-Model-FlowField-v1  
> **基准数据集**：The Well `shear_flow` (2D 不可压缩剪切流 + 被动示踪标量)  
> **计算环境**：NVIDIA vGPU-32GB × 2 (Ada Lovelace AD103 / RTX 4080 32G, CUDA 13.0, PyTorch 2.10.0+cu128，详见 [HARDWARE_ENVIRONMENT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/HARDWARE_ENVIRONMENT.md))  
> **工程测试基线**：全套自动化测试套件通过（256 项用例收集/回归测试全部绿灯通过，含协议契约、实验身份治理、H12 扩展契约、编译兼容性与时序对齐测试）

---

## 一、 整体研发路线图总览看板

| 阶段 | 核心任务 | 关键目标 / 验证标准 | 状态 |
| :--- | :--- | :--- | :--- |
| **Stage 1** | 数据基座与物理规范 | HDF5 数据审计、时间滑动窗口、物理参数归一化、测试集防泄漏协议（Sc 留出就绪，Re 留出因本地单参数标记为 BLOCKED_BY_DATA） | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 2** | 空间潜状态编码器与解码器 | 8x 下采样卷积结构、双向周期边界、压力零均值在评估阶段实施（解码器 project_pressure=False）、四场重建闭环 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 3** | 物理条件层与因子化 Transformer | $Re, Sc \to \text{log} \to \text{MLP} \to \text{AdaLN}$、时空解耦注意力、潜状态残差更新 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **基线体系** | 统一基线与公平评测竞技场 | Persistence、FNO-2D、Direct ST Transformer 接口契约统一 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 4.1** | 潜空间自由滚动机制 | `HistoryBuffer` 纯潜空间自回归推演、30 步滚动评测矩阵构建 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 4.2** | 双卡 DDP 短程自由滚动长训练 | 引入多步滚动监督 ($H=2$) 抑制自回归自激发散、双卡分布式训练 30 Epochs | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 4.3** | 自由滚动收敛复评 | 载入单步底座与滚动长训双模型展开 30 步评测，确认单步误差暴降 80.5% | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 5** | 物理守恒损失消融实验 (E0 - E4) | Closure-R4 物理损失 E0-E4 在三组种子（Seeds 42, 43, 44）上完成全量双卡训练、跨种子评测与配对检验闭环 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **Stage 6** | 周期 FFT 导数与独立物理评价系统 | 二维周期谱导数与拉普拉斯算子、全量物理衍生量、8 联排出版级 Rollout 看板导出 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **V1 全链验收** | 汇总主模型与基线、失败案例审计、架构规范与 V1 验收 | 实验身份契约全面冻结（P1-1~P1-5 闭环，Normalizer Hash 校验，Fail-Closed 阻断），Closure-R4 三种子消融与基线大盘汇总完成 | <font color="#2ea44f">● 已完成 (DONE)</font> |
| **10 月路线图** | 课程式长推演、潜流形扩散、宽域泛化与三维预研 | 课程式多步自回归展开（H=4, 8, 16）与 Pushforward 预热机制已落地通过回归测试；潜流形扩散与三维预研待调度 | <font color="#d29922">● 部分就绪 (IN PROGRESS)</font> |

---

## 二、 各阶段详细任务分解与执行证据

### Stage 1: 数据基座与物理规范
- [x] **HDF5 数据结构审计与元数据解析**：
  - 确认空间网格 $N_y \times N_x = 256 \times 512$，时间步长 $\Delta t = 1.0$；
  - 变量通道契约严格锁定为 $q = [u, v, p, s]$（四通道）；
  - *代码位置*：[src/data/shear_flow_dataset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/shear_flow_dataset.py)
- [x] **物理参数对数变换与归一化设计**：
  - Reynolds 数 $Re \in [10^4, 10^5]$、Schmidt 数 $Sc \in [0.1, 10.0]$；
  - 采用 $\log_{10}$ 映射至 $[-1, 1]$ 规范化空间；
  - 空间场采用仅在训练集划分上拟合的全局通道级均值方差归一化；引入 `split_hash` SHA-256 指纹校验，杜绝旧统计量静默复用；
  - *代码位置*：[src/data/normalization.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/normalization.py), [src/data/pipeline.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/pipeline.py)
- [x] **滑动窗口序列构建器**：
  - 支持历史观测窗口 $L$ 与预测窗口 $H$ 的自由滑动拼接，内置步长 `stride` 参数；
  - *代码位置*：[src/data/shear_flow_dataset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/shear_flow_dataset.py)
- [x] **严格防泄漏测试集划分协议**：
  - 实施 Trajectory-Level Grouped Split，同参数轨迹整体划分；
  - 支持 Schmidt 参数留出（`parameter_holdout_sc`）；
  - 本地单 Reynolds 数据下将 `parameter_holdout_re` 显式标记为 `BLOCKED_BY_DATA`，并在 Pipeline 中严格实施 Fail-Closed 阻断，拒绝空集虚假评测；
  - *代码位置*：[src/data/splits.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/data/splits.py), [outputs/splits/grouped_split.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/splits/grouped_split.json)

---

### Stage 2: 空间潜状态编码器与解码器
- [x] **2D 周期卷积空间编码器 (Encoder2D)**：
  - 采用自定义 `PeriodicConv2d`，保证 $x, y$ 双向圆周边界的严格连续性；
  - 3 层步长为 2 的周期卷积实现 $8\times$ 空间下采样压缩；
  - *代码位置*：[src/models/encoder.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/encoder.py)
- [x] **转置周期卷积空间解码器 (Decoder2D)**：
  - 对称上采样架构恢复全分辨率物理场；
  - 解码器遵循 `project_pressure=False`，压力场的物理零均值规范统一在物理评估阶段实施；
  - *代码位置*：[src/models/decoder.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/decoder.py)

---

### Stage 3: 物理条件嵌入与因子化时空 Transformer
- [x] **物理参数条件嵌入层 (ConditioningNetwork)**：
  - 支持 $Re, Sc$ 对数参数的高频傅里叶特征提取与 2 层 MLP 投影；
  - 导出 Scale 与 Shift 仿射向量用于 AdaLN-Zero 调制；
  - *代码位置*：[src/models/conditioning.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/conditioning.py)
- [x] **时空因子化注意力块 (SpatioTemporalBlock)**：
  - 空间自注意力与时序因果自注意力解耦计算；
  - 引入残差增量更新模式与零初始化输出门控；
  - *代码位置*：[src/models/latent_transformer.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/latent_transformer.py)

---

### 统一基线评测竞技场体系
- [x] **Persistence Baseline (B0)**：恒等惯性参考；
- [x] **FNO-2D Baseline (B1)**：傅里叶神经算子谱域卷积；
- [x] **Direct ST Transformer (B2)**：物理网格切片直接预测；
- [x] **基准契约统一验证**：在 `evaluate_rollout.py` 中引入 `verify_checkpoint_contract`，强校验参评各模型的数据分辨率、归一化与划分一致性；
- [x] *代码位置*：[src/baselines/fno.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/baselines/fno.py), [src/models/direct_transformer.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/direct_transformer.py), [scripts/evaluate_rollout.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_rollout.py)

---

### Stage 4: 纯潜空间自由滚动机制与多步展开
- [x] **纯潜空间 FIFO 滑动状态缓存 (HistoryBuffer)**：
  - 零误差内存级自回归推演；
  - *代码位置*：[src/models/history_buffer.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/history_buffer.py)
- [x] **双卡 DDP 短程自由滚动长训练 ($H=2$)**：
  - 引入多步滚动时序监督梯度回传，有效抑制单步自回归误差爆炸；
  - *代码位置*：[scripts/train_forecaster.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_forecaster.py)

---

### Stage 5: 物理守恒损失消融实验 (E0 - E4)
- [x] **消融评估体系升级至 E0~E4 五组**：
  - E0: Single-step pure field loss ($H=1, L_{\text{field}}$)；
  - E1: Rollout-aware field loss ($H=2, L_{\text{field}}$)；
  - E2: Divergence-free penalty ($H=2, +L_{\text{div}}$)；
  - E3: Vorticity-consistent penalty ($H=2, +L_\omega$)；
  - E4: Full physics coupling ($H=2, +L_{\text{div}} + L_\omega$)；
- [x] **Closure-R4 三种子（Seeds 42, 43, 44）全量重跑与配对检验**：
  - 双卡调度完成 15 组全量消融长训；
  - 产出出版级三种子统计检验汇总报告 [outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json) 与论文第 5 节成果；
  - *代码位置*：[scripts/run_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_physics_ablation.py), [scripts/evaluate_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_physics_ablation.py), [scripts/aggregate_multi_seed.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/aggregate_multi_seed.py)

---

### Stage 6: 周期 FFT 导数与独立物理评价系统
- [x] **周期快速傅里叶变换导数与微分算子库**：
  - 二维周期谱梯度、散度、涡量、拉普拉斯算子（解析解相对误差达机器极限 $1.40 \times 10^{-12}$）；
  - 动能、拟能与能谱分析；
  - 增加可微 Leray 谱投影算子 (`leray_projection_2d`)；
  - *代码位置*：[src/utils/fft_derivatives.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/utils/fft_derivatives.py)
- [x] **出版级 8 联排 Rollout 物理曲线看板生成器**：
  - [scripts/plot_rollout_comparison.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/plot_rollout_comparison.py)

---

### V1 阶段全链验收与失败案例审计
- [x] **测试轨迹逐条误差审计与失败案例分析**：
  - 消除静默 fallback 隐患，支持 `--allow_legacy_checkpoint` 显式 opt-in；
  - 导出典型对比大图与诊断 JSON 元数据；
  - *代码位置*：[scripts/analyze_failure_cases.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_failure_cases.py)
- [x] **系统架构说明与技术规范文档编写**：
  - [docs/ARCHITECTURE.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/ARCHITECTURE.md)
- [x] **V1 阶段全链验收报告与正式论文成果**：
  - [docs/V1_ACCEPTANCE_REPORT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/V1_ACCEPTANCE_REPORT.md)
  - [docs/MANUSCRIPT_RESULTS.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/MANUSCRIPT_RESULTS.md)

---

### 10 月份下一轮研发任务路线图 (October Roadmap)
- [x] **任务 1：课程式长时程多步自回归展开 ($H=4, 8, 16$) 与推前训练机制**：
  - 完成 `CurriculumRolloutScheduler`（倍增/线性/固定阶段调度）；
  - 完成 `HistoryBuffer` 与 `LatentForecaster` 的截断梯度推前预热机制（`stop-gradient pushforward`）；
  - 严格规范推前训练时间契约：`future` 模式监督 $q_{\text{future}}[K:K+H]$；`history` 模式底层支持扩展上下文切片，标准 4 步数据入口实施 fail-closed 安全隔离；
  - 完成 Horizon-R1（$H=2, 4, 8$）与 Horizon-R2（$H=16$ 双卡 DDP）长程推演训练与物理大盘评测；
  - 通过 16 项专用回归测试套件（`tests/test_curriculum_pushforward.py`）；
  - *代码位置*：[src/training/curriculum.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/training/curriculum.py), [scripts/train_forecaster.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_forecaster.py), [scripts/run_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_horizon_ablation.py), [scripts/run_h16_extension.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_h16_extension.py)
- [ ] **任务 2：潜流形生成式扩散世界模型 (Latent Diffusion Flow Model)**
- [ ] **任务 3：宽参数域泛化与极端工况外推适应性**
- [ ] **任务 4：三维不可压缩流场与复杂几何架构预研**

---

## 三、 硬件能效说明：显存占用与功率利用率

在实际多卡训练中，**显存占用约为 9GB / 32GB（未满），但 GPU 功率与计算利用率达到 90%~100%（满载）**：
1. **显存未占满**：编码器 $8\times$ 空间压缩使潜空间点数缩减 64 倍，极大降低长程自回归激活值缓存需求；
2. **功率与利用率打满**：多层 Transformer 多头注意力运算密集，双卡 DDP 与 AMP 调度饱满，处于最优 Compute-Bound 工况。
