# ADR-005: 概率潜流形方差建模与不确定性量化治理架构 (Probabilistic Latent Dynamics & Uncertainty Quantification)

## 状态 (Status)
**Accepted** (2026-09-27)

## 背景 (Context)
剪切流等湍流系统在物理本质上具有强烈的对流失稳（Kelvin-Helmholtz 不稳定性）与多尺度混沌特性。初始扰动的小微差在长时程自由滚动（Autoregressive Rollout）中会引发显著的偶然不确定性（Aleatoric Uncertainty）。
确定性模型输出单个点预测，在多步自回归中容易出现高频能谱虚假耗散（过度平滑）或误差累积发散；更重要的是，确定性模型无法提供预测结果的可信度置信区间（Confidence Intervals），难以作为可靠的世界模型（World Model）底座。

## 决策 (Decision)

我们设计并落地了四阶段闭环的概率潜流形动力学系统（ProbLatent Phase 0 ~ Phase 3）：

1. **Phase 0：潜转移残差审计与统计基准 (Audit & Contract Binding)**
   - 提取确定性 Transformer 在验证集上的单步潜转移残差 $\Delta Z = Z_{t+1} - Z_t$；
   - 统计逐通道与全网格的均值、方差及极值，锁定基准文件 `latent_residual_stats.json`，并由 `verify_latent_audit_contract.py` 实施 SHA-256 指纹强校验。

2. **Phase 1：均值-方差解耦与结构零误差平价 (Mean-Variance Decoupling & Structural Parity)**
   - 构建 `ProbabilisticLatentDynamics` 封装层；
   - 冻结确定性均值网络 $\mu_\theta(Z_t, c)$，接入轻量级空间对角高斯异方差头 $\sigma^2_\phi(Z_t, c)$；
   - 确立结构平价契约：提取均值或禁用扰动时，概率推演输出与原确定性底座严格保持 100% 数值一致（$\max |\mu_{\text{prob}} - \hat{Z}_{\text{det}}| \equiv 0.0$）。

3. **Phase 2：高斯负对数似然 (NLL) 损失与安全方差训练 (NLL Variance Training)**
   - 采用 PyTorch 原生 `F.gaussian_nll_loss` 进行异方差模型训练；
   - 设定方差保护下界 $\sigma_{\min}^2 = 10^{-4}$，防止数值除零；
   - 验证集 NLL 从同方差基线 G0 的 `0.0237` 显著优化至异方差模型 G1 的 `-0.2066`（提升 `-0.230 nats/element`）。

4. **Phase 3：自回归集合推演、区间校准与 Spread-Skill 评估 (Ensemble Rollout & Calibration)**
   - 支持蒙特卡洛重参数化采样自回归推演；
   - 实施滑动窗口 Cross-Window Pooled RMS Spread-Skill 跨窗口混合池化，消除跨时间窗口拼接的统计伪影；
   - 严格评估 50%、80%、90%、95% 名义区间的预测区间覆盖概率 (PICP) 与平均区间宽度 (MPIW)，实测 80% 区间覆盖率达 79.6%（绝对误差仅 0.4%），90% 区间覆盖率达 90.6%（绝对误差仅 0.6%）。

## 影响 (Consequences)

### 正面收益
- 赋予流场世界模型精确的不确定性度量能力，在保留确定性预测高精度的同时，提供了可信度指标；
- 测试集单步 NLL 从 `0.0270` 优化至 `-0.1704`，CRPS 改善 6.1%（0.3836 $\to$ 0.3602）；
- 产出了完整的出版级矢量 PDF/PNG 评估套件（Figure 1~3 与 Summary）。

### 权衡与约束
- 多步集合推演需要执行 $M$ 次自回归采样，推理计算开销随采样数线性增加；
- 方差头仅建模对角协方差（空间对角假设），未来可进一步探索非局部协方差或潜流形扩散生成模型。
