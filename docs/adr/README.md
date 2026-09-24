# 架构决策记录 (Architecture Decision Records, ADR)

本项目采用轻量级 ADR 规范记录关键系统架构与科学契约的设计决策、背景权衡与演进历程。

## 决策记录索引 (ADR Index)

| 编号 | 标题 | 状态 | 决策领域 | 核心影响与契约 |
| :--- | :--- | :--- | :--- | :--- |
| [ADR-001](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-001-latent-world-model-architecture.md) | 空间潜流形解耦与物理守恒梯度穿透架构 | **Accepted** | 模型架构 | 空间 8x 压缩流形、纯潜空间 FIFO 滚动、双向周期卷积、反归一化物理损失穿透 |
| [ADR-002](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-002-spatial-axis-ordering-and-spectral-derivatives.md) | 空间网格轴序契约与双向周期 FFT 谱导数算子 | **Accepted** | 物理与数据 | `[Ny, Nx] = [128, 256]` 严格维度对齐、Dedalus 数据集转置防护、二维实数傅里叶谱梯度 |
| [ADR-003](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-003-outputs-and-artifacts-governance.md) | 产物分层治理规范与符号链接向后兼容策略 | **Accepted** | 工程治理 | `outputs/{figures,logs,metrics}/` 模块化子目录、同名相对软链接保活、零破坏性回归保障 |
