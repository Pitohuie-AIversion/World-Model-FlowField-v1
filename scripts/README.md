# 流水线脚本库全景拓扑与调用指南 (Scripts Architecture & Workflow)

本目录包含流场世界模型（World-Model-FlowField-v1）完整科研闭环的 40 个核心脚本。涵盖数据工程、潜空间表征、确定性/概率动力学推演、物理先验消融、跨度阶梯自回归（H2/H4/H8/H12/H16）、多种子聚合检验、Navier-Stokes 与被动标量 PDE 动力学残差体系、机理诊断、超长程定性对比以及论文出版动态多媒体套件。

---

## 1. 实验流水线全景图 (Pipeline Architecture)

```
[ 1. 数据工程与物理契约校验 ]
  download_subset.py ──► build_splits.py ──► verify_splits.py
                                │
                                ├──► verify_schmidt_invariance.py / inspect_dataset.py
                                │
                                └──► visualize_dataset.py / visualize_fields.py
                                │
[ 2. 空间潜流形表示学习 (Stage B) ]
  train_representation.py (Encoder2D + Decoder2D, 8x 空间压缩)
                                │ (冻结解码器 / 提供可微梯度穿透)
                                ▼
[ 3. 时空动力学世界模型推演 (Stage C/D/H) ]
  ├── train_forecaster.py (单次单卡/单模型训练入口，支持编译加速与推前训练)
  ├── run_physics_ablation.py (Closure-R4 物理损失 E0-E4 双卡调度)
  ├── run_horizon_ablation.py (Horizon-R1 跨度 H2/H4/H8 消融调度)
  ├── run_h12_extension.py (Horizon-R2 H12 扩展调度器)
  └── run_h16_extension.py (Horizon-R2 极端长跨度 H16 双卡 DDP 加速)
                                │
                                ├───────────────────────────────┐
                                ▼                               ▼
[ 4. 概率潜流形动力学与不确定性量化 ]             [ 5. 偏微分方程 (PDE) 动力学残差受控训练 ]
  ├── compute_latent_statistics.py (潜残差审计)     ├── audit_pde_residuals.py (PDE 残差基线审计)
  ├── verify_latent_audit_contract.py (契约指纹)    ├── probe_pde_gradient_scales.py (梯度范数探测)
  ├── train_prob_latent_variance.py (方差头训练)    ├── run_pde_controlled_training.py (受控训练微调)
  ├── evaluate_prob_latent_phase3.py (自回归评测)   └── evaluate_pde_controlled_candidates.py (验证集基准评测)
  └── plot_prob_latent_phase3.py (出版矢量图/校准)
                                │
                                ▼
[ 6. 评测与统计聚合 (Evaluation & Multi-Seed) ]
  ├── evaluate_physics_ablation.py ──► aggregate_multi_seed.py
  ├── evaluate_horizon_ablation.py ──► (遴选 H8 Saved Long-Best 模型)
  ├── evaluate_h16_benchmark.py (H16 极限展开基准物理评测)
  └── evaluate_rollout.py (长程自回归基准对比)
                                │
                                ▼
[ 7. 动力学机理与极端案例诊断 (Deep Analysis) ]
  ├── analyze_training_convergence.py (损失收敛动力学分析)
  ├── analyze_spectral_dissipation.py (二维 FFT 能谱与拟能级联分析)
  └── analyze_failure_cases.py (长程推演失败案例与误差分位数诊断)
                                │
                                ▼
[ 8. 论文正文图表、多媒体视频与渲染 (Paper Artifacts & Media) ]
  ├── generate_paper_figures.py (论文核心图 Figure A & B)
  ├── generate_paper_tables.py (论文 Table 1 - 4 LaTeX 源码)
  ├── generate_qualitative_figures.py (流场定性比对与多时间步演化三联图)
  ├── visualize_h16_comparison.py (H16 多模型定性对比面板与 provenance 校验)
  ├── generate_shear_flow_video.py (200 帧完整时序 2x2 联动视频与 GIF 动画)
  ├── plot_physics_ablation.py (物理损失消融曲线绘制)
  └── plot_rollout_comparison.py (长程滚动基准对比图)
```

---

## 2. 脚本分类索引 (Functional Directory)

### 2.1 数据工程与流场校验 (Data Engineering & Preprocessing)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [download_subset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/download_subset.py) | 下载 The Well 剪切流数据集子集 | 从 HuggingFace 同步 HDF5 文件至 `data/` |
| [build_splits.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/build_splits.py) | 构建参数外推与分组划分 | 输出 `outputs/splits/*.json` |
| [verify_splits.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/verify_splits.py) | 验证划分样本无重叠与哈希一致性 | 校验数据契约 |
| [verify_schmidt_invariance.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/verify_schmidt_invariance.py) | 验证被动标量在不同 Schmidt 数下的物理标度 | 统计检验标量输运性质 |
| [inspect_dataset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/inspect_dataset.py) | 审计流场通道极值、均值与方差 | 输出流场审计报告 |
| [visualize_dataset.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/visualize_dataset.py) | 绘制原始流场样本与能谱诊断 | 生成流场状态切片 |
| [visualize_fields.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/visualize_fields.py) | 流场空间切片、流线与二维涡量分布渲染 | 生成高分辨率可视化切片 |

### 2.2 空间潜表示学习 (Representation Learning - Stage B)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [train_representation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_representation.py) | 训练 2D 卷积空间自编码器 (Encoder2D + Decoder2D) | 输出 `outputs/checkpoints/representation/best_vrmse_mean.pt` |

### 2.3 动力学世界模型推演与消融 (Dynamics Modeling & Horizon Ablations)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [train_forecaster.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_forecaster.py) | 动力学推演训练主入口 (单模型 / 单卡) | 训练 LatentForecaster 或 DirectSTTransformer |
| [run_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_physics_ablation.py) | Closure-R4 5组物理消融实验调度器 (E0-E4，支持双卡并行与多随机种子) | 输出日志至 `outputs/logs/closure_r4/` 与检查点 |
| [run_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_horizon_ablation.py) | Horizon-R1 推演跨度消融调度器 (H=2, 4, 8) | 输出日志至 `outputs/logs/horizon_r1/` 与检查点 |
| [run_h12_extension.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_h12_extension.py) | Horizon-R2 H12 扩展训练调度器 (双卡 DDP 加速) | 输出检查点与日志 |
| [run_h16_extension.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_h16_extension.py) | Horizon-R2 超长推演跨度 (H=16) 双卡 DDP 分布式加速训练 | 输出日志至 `outputs/logs/horizon_r2/` |
| [run_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_ablation.py) | 早期综合消融实验主调度脚本 | 早期探索性基准调度 |

### 2.4 概率潜空间动力学与不确定性量化 (ProbLatent Phase 0 ~ Phase 3)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [compute_latent_statistics.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/compute_latent_statistics.py) | 审计确定性底座的单步潜转移残差统计量 (均值/方差/极值) | 输出 `outputs/normalization/latent_residual_stats.json` |
| [verify_latent_audit_contract.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/verify_latent_audit_contract.py) | 校验潜残差审计契约完整性、哈希指纹与版本一致性 | 输出 `outputs/normalization/latent_audit_verification_record.json` |
| [train_prob_latent_variance.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_prob_latent_variance.py) | 训练潜空间对角高斯异方差头 (基于高斯负对数似然 NLL 损失) | 输出检查点及 `outputs/normalization/phase2_variance_training_record.json` |
| [evaluate_prob_latent_phase3.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_prob_latent_phase3.py) | 概率潜空间多步自回归推演、不确定性区间校准与 Spread-Skill 评估 | 输出 `outputs/metrics/phase3_probabilistic_evaluation.json` |
| [plot_prob_latent_phase3.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/plot_prob_latent_phase3.py) | 渲染出版级概率评估图表（VRMSE 演化、区间校准曲线、Spread-Skill 与概览） | 输出 `outputs/figures/probabilistic/` (PNG & 矢量 PDF) |

### 2.5 Navier-Stokes 与示踪剂 PDE 动力学残差受控训练体系

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [audit_pde_residuals.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/audit_pde_residuals.py) | 审计真值 (GT)、降采样场、自编码器与推演输出的 PDE 连续方程物理残差 | 输出 `outputs/evaluations/pde_residual_audit.json` |
| [probe_pde_gradient_scales.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/probe_pde_gradient_scales.py) | 探测 PDE 残差损失与场值损失对潜 Transformer 参数的梯度范数与余弦相似度 | 输出 `outputs/evaluations/pde_gradient_probe.json` |
| [run_pde_controlled_training.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_pde_controlled_training.py) | 调度 PDE 受控微调训练（对照组 P0 与实验组 PDE 受控训练） | 输出检查点及 `outputs/evaluations/pde_controlled_training_h12.json` |
| [evaluate_pde_controlled_candidates.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_pde_controlled_candidates.py) | 在全体验证集轨迹上全面评测 PDE 受控候选模型并计算相对增益 | 输出 `outputs/evaluations/pde_controlled_candidates_full_val.json` |

### 2.6 评测与统计聚合 (Evaluation & Statistical Aggregation)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [evaluate_rollout.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_rollout.py) | 长程多步自回归滚动性能评估 | 输出 `outputs/metrics/rollout_benchmark.json` |
| [evaluate_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_physics_ablation.py) | Closure-R4 物理损失消融全面评估 | 输出 `outputs/metrics/closure_r4_physics_ablation_*.json` |
| [evaluate_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_horizon_ablation.py) | Horizon-R1 跨度消融测试集评估与长程指标筛选 | 输出 `outputs/metrics/horizon_r1_test_evaluation.json` |
| [evaluate_h16_benchmark.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_h16_benchmark.py) | H16 极限展开实验物理基准全面评测 (涡量、散度、能谱综合评估) | 输出 H16 综合评测指标与诊断 |
| [aggregate_multi_seed.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/aggregate_multi_seed.py) | 聚合 Seed 42, 43, 44 评测指标并执行配对检验 | 输出 `outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json` |

### 2.7 深度物理分析与诊断 (Deep Diagnostics & Spectral Mechanics)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [analyze_training_convergence.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_training_convergence.py) | 解析训练日志，分析损失收敛轨迹与速率 | 输出 `training_convergence_summary.json` 与曲线图 |
| [analyze_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_horizon_ablation.py) | 跨度消融学习轨迹、验证集收敛与双指标遴选分析 | 输出 horizon_r1 统计轨迹并遴选 H8 最佳模型 |
| [analyze_spectral_dissipation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_spectral_dissipation.py) | 计算二维能谱、拟能级联与方向各向异性耗散比 | 输出 `directional_spectral_analysis.json` 与能谱比曲线 |
| [analyze_failure_cases.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_failure_cases.py) | 识别与分析最差推演轨迹与发散机制 | 输出 `failure_cases_analysis.json` 与故障诊断图 |

### 2.8 论文正文图表、多媒体视频与定性渲染 (Paper Artifacts, Media & Qualitative)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [generate_paper_figures.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_paper_figures.py) | 绘制论文核心高质量图表 (Figure A & B) | 输出 `outputs/figures/manuscript/figure_*.png` |
| [generate_paper_tables.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_paper_tables.py) | 自动提取指标生成学术论文 LaTeX 表格 (Table 1 - 4) | 输出 `outputs/tables/table_*.tex` |
| [generate_qualitative_figures.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_qualitative_figures.py) | 生成流场空间分布、基线对比与多步演化定性图 | 输出 `outputs/figures/qualitative/` 与对应 metadata |
| [visualize_h16_comparison.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/visualize_h16_comparison.py) | 严苛验证 Provenance 并生成 H16 多模型超长程演化定性对比面板 | 输出 `outputs/figures/h16_comparison/` 与元数据索引 |
| [generate_shear_flow_video.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_shear_flow_video.py) | 渲染 200 帧完整剪切流生命周期时序四联动态视频与预览 GIF | 输出 `outputs/videos/*.mp4` 与 `*.gif` |
| [plot_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/plot_physics_ablation.py) | 绘制单种子与多种子物理消融多指标对比曲线 | 输出 `outputs/figures/closure_r4/` 曲线图 |
| [plot_rollout_comparison.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/plot_rollout_comparison.py) | 绘制长程自回归滚动推演对比基线误差曲线 | 输出 `outputs/figures/benchmark/rollout_benchmark_curves.png` |

---

## 3. 典型实验操作范例 (Reproducible Command Recipes)

### 3.1 运行 Closure-R4 物理消融实验 (双卡加速)
```bash
# 执行完整 5 组 (E0-E4) 消融训练 (自动调度 GPU 0/1)
python scripts/run_physics_ablation.py --seed 42

# 全量评测物理消融模型
python scripts/evaluate_physics_ablation.py --seed 42 --checkpoint_tag v2

# 多种子聚合统计 (在完成 seed 42, 43, 44 后执行)
python scripts/aggregate_multi_seed.py
```

### 3.2 运行 Horizon 跨度消融与长程选型
```bash
# 调度 H2, H4, H8 跨度训练
python scripts/run_horizon_ablation.py --seed 42

# 调度 H12 扩展训练
python scripts/run_h12_extension.py --seed 42

# 评测长程泛化性能并选择最佳检查点
python scripts/evaluate_horizon_ablation.py --seed 42

# 生成定性可视化流场三联图 (基于长程最佳 H8 模型)
python scripts/generate_qualitative_figures.py --preset h8_long_eval
```

### 3.3 运行概率潜空间动力学全流程 (ProbLatent Phase 0 ~ Phase 3)
```bash
# Phase 0: 潜空间残差均值与方差审计
python scripts/compute_latent_statistics.py
python scripts/verify_latent_audit_contract.py

# Phase 2: 训练对角高斯异方差网络
python scripts/train_prob_latent_variance.py --seed 42 --epochs 5

# Phase 3: 自回归推演、不确定性区间校准与 Spread-Skill 评测
python scripts/evaluate_prob_latent_phase3.py --seed 42

# Phase 3: 渲染高分辨率出版级矢量 PDF 与 PNG
python scripts/plot_prob_latent_phase3.py
```

### 3.4 运行 Navier-Stokes & Tracer PDE 动力学残差与受控训练
```bash
# 审计 GT、自编码器与推演模型的 PDE 残差基线
python scripts/audit_pde_residuals.py

# 探测 PDE 物理损失相对场损失的梯度范数与相似度
python scripts/probe_pde_gradient_scales.py

# 运行受控微调实验 (P0 对照组 vs PDE 实验组)
python scripts/run_pde_controlled_training.py --steps 50

# 在全体验证集上全面评测候选模型
python scripts/evaluate_pde_controlled_candidates.py
```

### 3.5 渲染 200 帧生命周期全景动态视频与 H16 定性比对
```bash
# 渲染 200 帧 2x2 四联全景视频 (20 FPS, 10.0 秒)
python scripts/generate_shear_flow_video.py \
  --hdf5_path /root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5 \
  --sim_idx 0 \
  --layout quad \
  --output_video outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames.mp4

# 生成 H16 对比图表并严格校验样本溯源契约
python scripts/visualize_h16_comparison.py
```


