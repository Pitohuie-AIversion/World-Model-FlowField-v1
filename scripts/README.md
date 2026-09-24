# 流水线脚本库全景拓扑与调用指南 (Scripts Architecture & Workflow)

本目录包含流场世界模型（World-Model-FlowField-v1）完整科研闭环的 27 个核心脚本。涵盖数据工程、潜空间表征、动力学推演训练、物理消融实验、多种子评测统计、动力学机理诊断及论文出版图表渲染。

---

## 1. 实验流水线全景图 (Pipeline Architecture)

```
[ 1. 数据工程与物理契约校验 ]
  download_subset.py ──► build_splits.py ──► verify_splits.py
                                │
                                └──► verify_schmidt_invariance.py / inspect_dataset.py
                                │
                                └──► visualize_dataset.py / visualize_fields.py
                                │
[ 2. 空间潜流形表示学习 (Stage B) ]
  train_representation.py (Encoder2D + Decoder2D, 8x 空间压缩)
                                │ (冻结/提供解码器雅可比穿透)
                                ▼
[ 3. 时空动力学世界模型推演 (Stage C/D/H) ]
  ├── train_forecaster.py (单次单卡/单模型训练入口)
  ├── run_physics_ablation.py (Closure-R4 物理损失 E0-E4 双卡调度)
  ├── run_horizon_ablation.py (Horizon-R1 跨度 H2/H4/H8 消融调度)
  └── run_h16_extension.py (Horizon-R2 极端长跨度 H16 双卡 DDP 加速)
                                │
                                ▼
[ 4. 评测与统计聚合 (Evaluation & Multi-Seed) ]
  ├── evaluate_physics_ablation.py ──► aggregate_multi_seed.py
  ├── evaluate_horizon_ablation.py ──► (遴选 H8 Saved Long-Best 模型)
  └── evaluate_rollout.py (长程自回归基准对比)
                                │
                                ▼
[ 5. 动力学机理与极端案例诊断 (Deep Analysis) ]
  ├── analyze_training_convergence.py (损失收敛动力学分析)
  ├── analyze_spectral_dissipation.py (二维 FFT 能谱与拟能级联分析)
  └── analyze_failure_cases.py (长程推演失败案例与误差分位数诊断)
                                │
                                ▼
[ 6. 论文正文图表与 LaTeX 表格渲染 (Paper Artifacts) ]
  ├── generate_paper_figures.py (论文核心图 Figure A & B)
  ├── generate_paper_tables.py (论文 Table 1 - 4 LaTeX 源码)
  ├── generate_qualitative_figures.py (流场定性比对与多时间步演化三联图)
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

### 2.3 动力学世界模型推演与消融 (Dynamics Modeling & Ablations - Stage C/D/H)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [train_forecaster.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_forecaster.py) | 动力学推演训练主入口 (单模型 / 单卡) | 训练 LatentForecaster 或 DirectSTTransformer |
| [run_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_physics_ablation.py) | Closure-R4 5组物理消融实验调度器 (E0-E4，支持双卡并行与多随机种子) | 输出日志至 `outputs/logs/closure_r4/` 与检查点 |
| [run_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_horizon_ablation.py) | Horizon-R1 推演跨度消融调度器 (H=2, 4, 8) | 输出日志至 `outputs/logs/horizon_r1/` 与检查点 |
| [run_h16_extension.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_h16_extension.py) | Horizon-R2 超长推演跨度 (H=16) 双卡 DDP 分布式加速训练 | 输出日志至 `outputs/logs/horizon_r2/` |
| [run_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/run_ablation.py) | 早期综合消融实验主调度脚本 | 早期探索性基准调度 |

### 2.4 评测与统计聚合 (Evaluation & Statistical Aggregation)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [evaluate_rollout.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_rollout.py) | 长程多步自回归滚动性能评估 | 输出 `outputs/metrics/rollout_benchmark.json` |
| [evaluate_physics_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_physics_ablation.py) | Closure-R4 物理损失消融全面评估 | 输出 `outputs/metrics/closure_r4_physics_ablation_*.json` |
| [evaluate_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_horizon_ablation.py) | Horizon-R1 跨度消融测试集评估与长程指标筛选 | 输出 `outputs/metrics/horizon_r1_test_evaluation.json` |
| [evaluate_h16_benchmark.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/evaluate_h16_benchmark.py) | H16 扩展实验物理基准全面评测 (涡量、散度、能谱综合评估) | 输出 H16 综合评测指标与诊断 |
| [aggregate_multi_seed.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/aggregate_multi_seed.py) | 聚合 Seed 42, 43, 44 评测指标并执行配对检验 | 输出 `outputs/metrics/closure_r4_physics_ablation_tri_seed_summary.json` |

### 2.5 深度物理分析与诊断 (Deep Diagnostics & Spectral Mechanics)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [analyze_training_convergence.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_training_convergence.py) | 解析训练日志，分析损失收敛轨迹与速率 | 输出 `training_convergence_summary.json` 与曲线图 |
| [analyze_horizon_ablation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_horizon_ablation.py) | 跨度消融学习轨迹、验证集收敛与双指标遴选分析 | 输出 horizon_r1 统计轨迹并遴选 H8 最佳模型 |
| [analyze_spectral_dissipation.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_spectral_dissipation.py) | 计算二维能谱、拟能级联与方向各向异性耗散比 | 输出 `directional_spectral_analysis.json` 与能谱比曲线 |
| [analyze_failure_cases.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/analyze_failure_cases.py) | 识别与分析最差推演轨迹与发散机制 | 输出 `failure_cases_analysis.json` 与故障诊断图 |

### 2.6 论文正文图表与 LaTeX 表格渲染 (Paper Artifacts & Tables)

| 脚本文件 | 功能说明 | 核心输入 / 输出 |
| :--- | :--- | :--- |
| [generate_paper_figures.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_paper_figures.py) | 绘制论文核心高质量图表 (Figure A & B) | 输出 `outputs/figures/manuscript/figure_*.png` |
| [generate_paper_tables.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_paper_tables.py) | 自动提取指标生成学术论文 LaTeX 表格 (Table 1 - 4) | 输出 `outputs/tables/table_*.tex` |
| [generate_qualitative_figures.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/generate_qualitative_figures.py) | 生成流场空间分布、基线对比与多步演化定性图 | 输出 `outputs/figures/qualitative/` 与对应 metadata |
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

# 评测长程泛化性能并选择最佳检查点
python scripts/evaluate_horizon_ablation.py --seed 42

# 生成定性可视化流场三联图 (基于长程最佳 H8 模型)
python scripts/generate_qualitative_figures.py --preset h8_long_eval
```

### 3.3 生成论文出版级成果物 (Manuscript Artifacts)
```bash
# 渲染正文核心 Figure A (VRMSE/方差耗散) 与 Figure B (物理守恒量)
python scripts/generate_paper_figures.py

# 计算能谱与方向各向异性耗散比曲线
python scripts/analyze_spectral_dissipation.py

# 生成正文全部 LaTeX 表格 (Table 1 - Table 4)
python scripts/generate_paper_tables.py
```

### 3.4 高性能编译加速训练 (PyTorch 2.x torch.compile)
```bash
# 开启内核融合加速 (TorchInductor 自动融合注意力与解码层，零显存碎片)
python scripts/train_forecaster.py --model latent_transformer --horizon 4 --compile
```

