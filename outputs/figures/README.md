# 可视化图表目录索引与管理规范 (Outputs Figures)

本目录存储模型评估、消融实验、论文撰写及定性分析所生成的所有可视化图表与对应元数据 JSON。

## 1. 目录结构概览

为了提升成果文件的模块化与检索效率，图表按照实验阶段与展示目的划分为以下子目录：

```text
outputs/figures/
├── README.md                      # 本说明文档
├── manuscript/                    # 论文正文与报告核心图表
│   ├── figure_a_vrmse_and_dispersion.png
│   ├── figure_b_physical_invariants.png
│   └── directional_spectral_ratio_curves.png
├── horizon_r1/                    # Horizon-R1 跨度消融实验诊断图表
│   ├── horizon_r1_comparison.png
│   └── horizon_r1_checkpoint_selection.png
├── closure_r4/                    # Closure-R4 物理损失与约束消融曲线
│   ├── closure_r4_physics_ablation_v2_curves.png
│   ├── closure_r4_physics_ablation_seed43_v2_curves.png
│   ├── closure_r4_physics_ablation_seed44_v2_curves.png
│   ├── closure_r4_physics_ablation_tri_seed_curves.png
│   ├── physics_ablation_curves.png
│   └── training_convergence_curves.png
├── benchmark/                     # 长程基准评测与异常诊断
│   ├── rollout_benchmark_curves.png
│   ├── failure_cases_analysis.png
│   └── flow_state_real.png
├── qualitative/                   # 定性流场空间分布与涡量场可视化
│   ├── horizon_r1/                # Horizon-R1 选型定性评估 (包含 H8 长程最佳模型对比)
│   │   ├── compare_parent_vs_h8_seed42_h30_u.png (+ metadata.json)
│   │   ├── h8_saved_long_best_seed42_h30_u_panel.png (+ metadata.json)
│   │   └── h8_saved_long_best_seed42_multistep_evolution_u.png (+ metadata.json)
│   └── [symlinks]                 # 指向 horizon_r1/ 的向后兼容相对软链接
└── [symlinks]                     # 根目录保留指向各模块的向后兼容相对软链接
```

## 2. 向后兼容性保障机制 (Symlink Compatibility)

所有迁移至子目录的文件均在原有扁平路径（`outputs/figures/*.png` 及 `outputs/figures/qualitative/*`）建立了**同名相对软链接**。
- **自动化测试兼容**：现有自动化测试脚本（如 `tests/test_horizon_ablation.py`、`tests/test_qualitative_figures.py`）无需做路径适配即可直接运行通过。
- **引用兼容**：Markdown 报告与实验文档中的历史链接保持有效，不破坏已有超链接生态。

## 3. 生成与复现命令

- **论文核心图生成**：
  ```bash
  python scripts/plot_manuscript_figures.py
  python scripts/plot_directional_spectral_ratio.py
  ```
- **Horizon-R1 消融图生成**：
  ```bash
  python scripts/analyze_horizon_ablation.py
  ```
- **Closure-R4 物理消融图生成**：
  ```bash
  python scripts/plot_closure_ablation.py
  ```
- **定性流场图 (Qualitative) 生成**：
  ```bash
  # H8 长程模型评估三联组图 (自动生成 PNG 与 metadata JSON)
  python scripts/generate_qualitative_figures.py --preset h8_long_eval
  ```
