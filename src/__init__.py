# World-Model-FlowField-v1

[World-Model-FlowField-v1](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1) 是流场物理世界模型 V1 实验库。

## 目录结构

- `configs/`: 数据、模型、训练与实验配置文件
- `src/data/`: 数据集加载、归一化、划分与时序滑动窗口
- `src/models/`: 空间编码器/解码器、条件注入、潜空间/直接 Transformer
- `src/losses/`: 场值损失、短程自由滚动损失、散度与涡量物理损失
- `src/metrics/`: 场值误差、谱分析、示踪标量、物理守恒与计算开销指标
- `src/baselines/`: Persistence、FNO 等基线模型
- `src/utils/`: 周期 FFT 导数、检查点管理与可复现性工具
- `scripts/`: 数据审计、划分构建、表示训练、动力学训练与评价脚本
- `tests/`: 自动化单元测试与物理性质验证测试
- `outputs/`: 权重保存、指标记录与可视化图表
