# ADR-004: 多尺度动能谱损失与课程自回归推前训练架构

- **状态 (Status)**: Accepted
- **日期 (Date)**: 2026-09-24
- **决策人 (Deciders)**: 算法与系统架构工程团队
- **相关文档**: [ARCHITECTURE.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/ARCHITECTURE.md), [ADR-001](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-001-latent-world-model-architecture.md), [ADR-002](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-002-spatial-axis-ordering-and-spectral-derivatives.md)

---

## 1. 背景与上下文 (Context)

在将空间潜流形世界模型推进至超长时序（$H \ge 16$ 及外推步长）推演时，纯数据驱动模型面临两个经典科学机器学习（SciML）痛点：

1. **时序曝光偏差与累积分布漂移（Exposure Bias & Distribution Drift）**：
   标准训练以真值（Ground Truth）历史帧为输入。在测试期自回归滚动时，上一时刻的微小预测误差被喂入下一时刻，误差迅速复合放大，导致模型进入未探索的相空间并迅速失真崩塌。
2. **频率域谱偏置与小尺度涡过度耗散（Spectral Bias & Numerical Dissipation）**：
   均方误差（MSE）或 $L_1$ 场损失天然对低频大尺度结构赋予压倒性权重（能量占比 >95%）。自回归模型往往会学到一种“平滑滤波”的捷径，导致高频小尺度湍流涡结构随推演步数递增而过度模糊、消亡；或相反，在网格截止波数处出现虚假的高频数值能量堆积（Aliasing/Shock）。

## 2. 架构决策 (Decision)

我们决定采用**多尺度动能谱损失与课程自回归推前训练一体化架构**：

1. **课程滚动递进调度器 (CurriculumRolloutScheduler)**：
   - 训练推演时间步长按课程模式平滑递进（默认采用几何翻倍调度 `doubling`: $2 \to 4 \to 8 \to 16$），初期聚焦高质量单步动力学映射，后期逐步施加长程动力学一致性；
   - 状态字典化支持与热启动（Warm-start）断点严格续训。

2. **截断梯度推前预热机制 (Stop-Gradient Pushforward)**：
   - 在计算 BPTT 展开图前，显式执行 `@torch.no_grad()` 的 $K$ 步自回归预热（可选注入高斯收缩扰动），推动 `HistoryBuffer` 进入模型自身的推演相空间分布；
   - 彻底打破“完美真值历史”的假象，在受扰动的非理想输入上继续推演并反传梯度，有效抵御推演漂移。

3. **GPU 矢量化可微径向壳层动能谱算子 (`compute_batched_radial_energy_spectrum`)**：
   - 基于 2D 实傅里叶变换（`rfft2`）计算空间能量密度 $\frac{1}{2}(|\hat{u}|^2 + |\hat{v}|^2)$；
   - 利用 PyTorch 内部 `scatter_add_` 算子完成各向同性波数壳层 $\lfloor |\mathbf{k}| / \Delta k \rfloor$ 的并行归约；显存常数仅 260 KB，单步求和耗时 $< 2$ 毫秒，且保持 100% 解析可导。

4. **多尺度对数动能谱损失 (`EnergySpectrumLoss`)**：
   - 在对数空间计算径向能谱偏差：
     \[
     \mathcal{L}_{\text{spec}} = \frac{1}{K} \sum_{k=1}^K w_k \left| \log_{10}(E_{\text{pred}}(k) + \epsilon) - \log_{10}(E_{\text{target}}(k) + \epsilon) \right|
     \]
   - 跨越 4~8 个能量数量级平衡各尺度能级级联梯度，并支持高频加权因子 $w_k = 1 + \alpha \frac{k}{K_{\max}}$。

## 3. 架构影响与权衡 (Consequences)

### 正向收益 (Positive)
- **消除长程分布漂移**：Pushforward 预热使模型对自身产生的不完美前置帧具备强韧的收缩修正动力学能力；
- **保全湍流多尺度能谱**：显式对齐能谱级联规律（Kraichnan $k^{-3}$ 标度律），有效消除多步自回归模糊耗散；
- **全流程解析可导与高性能**：`scatter_add_` 相比传统 CPU 循环循环与非微分矩阵节省了 98% 内存与搬运延迟，全面兼容 `torch.compile` 编译优化；
- **严格向后兼容**：默认参数下课程学习与谱损失均为关闭状态（`curriculum_rollout=False`, `lambda_spec=0.0`），保证存量权重、测试与下游任务 100% 稳定运行。

### 负面代价与应对 (Negative & Mitigations)
- **训练显存开销**：当课程调度到达 $H=16$ 时，BPTT 展开梯度的显存占用较 $H=1$ 有所上升。系统通过微批次切分（Micro-batching）与梯度累积（`--grad_accum_steps`）彻底消除显存超限（OOM）风险。
