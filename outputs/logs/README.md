# 实验训练日志目录索引与规范 (Outputs Logs)

本目录存储模型训练、消融实验与扩展实验的标准输出日志文件。

## 1. 目录结构概览

```text
outputs/logs/
├── README.md                      # 本说明文档
├── closure_r4/                    # Closure-R4 物理损失与约束消融训练日志 (Tri-Seed: 42, 43, 44)
│   ├── train_closure_r4_ablation_E0_single_step.log               # Seed 42 Baseline
│   ├── train_closure_r4_ablation_E1_rollout_field.log              # Seed 42 Rollout H=2
│   ├── train_closure_r4_ablation_E2_plus_L_div.log                 # Seed 42 + Div Penalty
│   ├── train_closure_r4_ablation_E3_plus_L_vort.log                # Seed 42 + Vort Penalty
│   ├── train_closure_r4_ablation_E4_full_physics.log               # Seed 42 Full Physics
│   ├── train_closure_r4_seed_43_ablation_E1_rollout_field.log      # Seed 43
│   ├── train_closure_r4_seed_43_ablation_E2_plus_L_div.log
│   ├── train_closure_r4_seed_43_ablation_E3_plus_L_vort.log
│   ├── train_closure_r4_seed_43_ablation_E4_full_physics.log
│   ├── train_closure_r4_seed_44_ablation_E1_rollout_field.log      # Seed 44
│   ├── train_closure_r4_seed_44_ablation_E2_plus_L_div.log
│   ├── train_closure_r4_seed_44_ablation_E3_plus_L_vort.log
│   └── train_closure_r4_seed_44_ablation_E4_full_physics.log
├── horizon_r1/                    # Horizon-R1 跨度消融训练日志 (Seed 42)
│   ├── train_horizon_r1_seed_42_E4_H2_control.log                 # H=2 Control Baseline
│   ├── train_horizon_r1_seed_42_E4_H4.log                         # H=4 Rollout
│   └── train_horizon_r1_seed_42_E4_H8.log                         # H=8 Rollout (Selected Long-Best)
├── horizon_r2/                    # Horizon-R2 极端长跨度双卡 DDP 加速扩展日志
│   └── seed_42/
│       └── E4_H16.log                                             # H=16 Rollout (Dual GPU DDP)
└── early_ablation/                # 早期探索性消融训练日志
    ├── train_ablation_A1_direct.log                               # Direct ST Transformer
    ├── train_ablation_A2_joint.log                                # Joint Fine-Tuning
    └── train_ablation_E0_single_step.log ... train_ablation_E4_full_physics.log
```

## 2. 向后兼容性保障机制 (Symlink Compatibility)

为了保证已有自动化测试（如 `tests/test_training_convergence.py`）与分析脚本（`scripts/analyze_training_convergence.py`）的正常运行：
- 在 `outputs/` 根目录下均保留了指向相应子目录同名日志文件的**相对软链接**；
- 任何现有脚本或测试通过 `outputs/train_*.log` 读取日志的行为完全保持不变。
