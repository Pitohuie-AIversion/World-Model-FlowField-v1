# ADR-007: 潜空间最优传输连续流匹配架构 (Latent Optimal Transport Conditional Flow Matching, OT-CFM)

## 状态 (Status)
**Accepted** (2026-09-29)

## 背景 (Context)
在流体世界模型 ProbLatent-R1 Phase 0 ~ Phase 3 的系统评测中，基于对角高斯异方差头（`VarianceHead2D`, $G_1$）的概率建模取得了显著进展（测试集 NLL 下降 $0.197\text{ nats/element}$，CRPS 相对改善 $6.11\%$），但暴露了三个根本性的物理与数学局限（参见 [PROB_LATENT_EVALUATION_REPORT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/PROB_LATENT_EVALUATION_REPORT.md)）：

1. **空间对角独立性破缺**：对角高斯独立白噪声破坏了不可压缩流体的椭圆型约束（$\nabla \cdot \mathbf{u} = 0$），导致解码后的物理速度散度从真实场 $0.003$ 暴增至 $1.918$；
2. **单峰对称性破缺**：高雷诺数剪切流动存在显著的 Kelvin-Helmholtz 失稳和涡破碎间歇性，残差分布呈现明显的尖峰、重尾甚至双稳态分叉，单峰钟形高斯无法正确覆盖，导致 50% 名义区间经验覆盖率仅有 $26.51\%$；
3. **随机扰动推离吸引子流形**：白噪声在多步滚动中将潜状态推向高维球壳，致使 30 步系综均值 VRMSE 较确定性基线退化 $23.15\%$。

为了从根本上解决上述局限，对标 Google DeepMind 的 **GenCast** (Nature 2024) 与 **SEEDS** (Science 2024) 的前沿范式，需要将流体世界模型推进至**潜空间流匹配 (Latent Flow Matching / OT-CFM)**。

## 决策 (Decision)

我们设计并落地了潜空间最优传输连续归一化流系统（`LatentFlowMatcher`）：

1. **冻结底座与残差流匹配 (Residual Latent OT-CFM)**：
   - 保持确定性主干网络 $D_0$（`LatentSTTransformer`）与空间自编码器（`Encoder2D`, `Decoder2D`）完全冻结（`requires_grad=False`）；
   - 将流匹配的目标分布定义在以确定性预测均值为中心的潜转移残差上：$x_1 = r_t = Z_{t+1} - \mu_t \in \mathbb{R}^{B \times C_z \times H_z \times W_z}$；
   - 先验基分布为标准高斯：$x_0 \sim \mathcal{N}(0, I)$；
   - 最优传输概率路径：$x_\tau = (1 - (1 - \sigma_{\min})\tau) x_0 + \tau x_1$，对应恒定条件速度场 $u_\tau(x_\tau \mid x_0, x_1) = x_1 - (1 - \sigma_{\min})x_0$。

2. **二维双向周期性残差速度网络 (`LatentVelocityNet2D`)**：
   - **时间嵌入**：连续流时间 $\tau \in [0, 1]$ 经 `SinusoidalTimeEmbedding` 正弦/余弦频域投影与 MLP 映射；
   - **多物理条件注入**：将时间嵌入与 $(Re, Sc)$ 物理参数嵌入拼接，经 AdaLN 调制注入到各个卷积残差块中；
   - **周期性物理对称性**：全部卷积层采用 `padding_mode="circular"`（双向周期性边界填充），严格维持剪切流动在潜流形上的环面拓扑不变性；
   - **空间相干性建模**：利用空间残差卷积块与轻量级潜空间自注意力机制（`LatentSpatialAttention2D`），显式捕获非局部涡旋相干结构，从根本上克服白噪声带来的高频噪点；
   - **零初始化与平价契约**：速度输出层（`Conv2d`）采用零初始化（Zero-Init）。在零基噪声或未训练时，速度场恒为 0，采样严格退化为确定性主干预测 $\mu_t$（$\max |Z_{\text{sample}} - \mu_t| \equiv 0.0$）。

3. **高阶数值 ODE 积分求解器 (`ODESolver`)**：
   - 实现包含 `euler`（一阶）、`midpoint`（二阶）、`heun`（二阶）及 `rk4`（四阶 Runge-Kutta）的高精度并行数值积分器；
   - 支持动态可配置积分步数（如 $N \in [5, 20]$），平滑权衡推断速度与生成质量；
   - 支持 `noise_scale` 控制系数（$\text{noise\_scale}=0.0$ 时确定性平价输出，$\text{noise\_scale}=1.0$ 时全概率采样）。

4. **端到端世界模型集成 (`LatentForecaster`)**：
   - 增加 `attach_flow_matcher()`、`freeze_for_flow_matching_training()` 与 `sample_rollout_flow_matching()`；
   - 支持单源多轨迹并行滚动（Single-Source Multi-Trajectory Rollout），通过批次 $K$ 倍扩展保证各采样轨迹独立的自回归历史缓冲，杜绝跨轨迹污染；
   - 严格向下兼容，原有确定性接口与方差头接口无任何破坏。

## 影响 (Consequences)

### 正向收益 (Positive)
- **空间相干性突破**：从白噪声采样演进为基于连续流形演化的相干流场生成，杜绝非物理高频噪点；
- **多模态与重尾建模**：流匹配作为无先验形态约束的连续归一化流，天然具备拟合剪切失稳多模态分叉与间歇性尖峰的能力；
- **严格平价与平滑回退**：零初始化与残差建模确保模型在训练初始阶段或噪声关闭时 100% 还原最优确定性预测 $D_0$；
- **完备的工程治理**：配套完整的单元测试、契约测试与严格校验训练脚本。

### 权衡与约束 (Trade-offs & Constraints)
- **推断计算开销**：单步采样需执行 $N$ 次速度网络评估（例如 10 步 Midpoint 积分），计算量高于单次高斯采样，但得益于潜空间尺寸极小（$16 \times 16 \times 64$），单步仍可在毫秒级完成。
