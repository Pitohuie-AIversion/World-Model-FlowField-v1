# ADR-007: 潜空间条件流匹配架构 (Residual Latent Conditional Flow Matching, Latent CFM)

## 状态 (Status)
**Accepted** (2026-09-29)

## 背景 (Context)
在流体世界模型 ProbLatent-R1 Phase 0 ~ Phase 3 的系统评测中，基于对角高斯异方差头（`VarianceHead2D`, $G_1$）的概率建模取得了显著进展（测试集 NLL 下降 $0.197\text{ nats/element}$，CRPS 相对改善 $6.11\%$），但暴露了三个根本性的物理与数学局限（参见 [PROB_LATENT_EVALUATION_REPORT.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/PROB_LATENT_EVALUATION_REPORT.md)）：

1. **空间对角独立性破缺**：对角高斯独立白噪声破坏了不可压缩流体的空间连续性与椭圆型约束（$\nabla \cdot \mathbf{u} = 0$），导致解码后的物理速度散度从真实场 $0.003$ 暴增至 $1.918$；
2. **单峰对称性破缺**：高雷诺数剪切流动存在显著的 Kelvin-Helmholtz 失稳和涡破碎间歇性，残差分布呈现明显的尖峰、重尾甚至双稳态分叉，单峰钟形高斯无法正确覆盖，导致 50% 名义区间经验覆盖率仅有 $26.51\%$；
3. **随机扰动推离吸引子流形**：白噪声在多步滚动中将潜状态推向高维球壳，致使 30 步系综均值 VRMSE 较确定性基线退化 $23.15\%$。

### 学术定位与范式溯源
- **思路启发**：受到 Google DeepMind **GenCast** (Nature 2024) 与 **SEEDS** (Science 2024) 生成式系综预报思路的启发，但 GenCast 与 SEEDS 本质上采用的是扩散模型 (Diffusion Models, DDPM/EDM)；
- **算法架构对标**：本项目在算法架构与数学实现上更直接对标 **ArchesWeatherGen (Science Advances 2024)**：采用 **“确定性基座预测 + 尺度归一化残差 + 连续流匹配生成 + 自回归系综推演”** 的两阶段残差生成流模型范式；
- **路径命名规范**：本项目采用 Lipman 等人的 conditional OT Gaussian 概率路径（条件最优传输直移插值），未引入 Tong 等人的全批次全局 OT coupling，因此科学命名严格限定为 **Residual Latent Conditional Flow Matching (Latent CFM)**。

## 决策 (Decision)

我们设计并落地了潜空间条件连续流系统（`LatentFlowMatcher`）：

1. **冻结底座与残差流匹配 (Residual Latent CFM)**：
   - 保持确定性主干网络 $D_0$（`LatentSTTransformer`）与空间自编码器（`Encoder2D`, `Decoder2D`）完全冻结（`requires_grad=False`）；
   - **残差尺度归一化 (Residual Scale Normalization, 对齐 ArchesWeatherGen)**：基于 Phase 0 计算的 64 通道二阶矩统计量 $s_c = \sqrt{\mathbb{E}[r_c^2] + \epsilon}$，将原始残差归一化至单位尺度：
     \[ \tilde{r} = \frac{Z_{t+1} - \mu_t}{s_c} \]
     流匹配在归一化残差空间中训练，彻底消除各潜通道方差跨度 30 倍导致的均方误差权重倾斜；采样后通过 $\hat{r} = \hat{\tilde{r}} \odot s_c$ 复原；
   - 先验基分布为标准高斯：$x_0 \sim \mathcal{N}(0, I)$；
   - 直移概率路径：$x_\tau = (1 - (1 - \sigma_{\min})\tau) x_0 + \tau x_1$，对应恒定条件速度场 $u_\tau(x_\tau \mid x_0, x_1) = x_1 - (1 - \sigma_{\min})x_0$。

2. **二维双向周期性残差速度网络 (`LatentVelocityNet2D`)**：
   - **时间嵌入**：连续流时间 $\tau \in [0, 1]$ 经 `SinusoidalTimeEmbedding` 正弦/余弦频域投影与 MLP 映射；
   - **多物理条件注入**：将时间嵌入与 $(Re, Sc)$ 物理参数嵌入拼接，经 AdaLN 调制注入到各个卷积残差块中；
   - **周期性物理对称性**：全部卷积层采用 `padding_mode="circular"`（双向周期性边界填充），严格维持剪切流动在潜流形上的环面拓扑不变性；
   - **空间相干性建模**：利用空间残差卷积块与轻量级潜空间自注意力机制（`LatentSpatialAttention2D`），显式捕获非局部涡旋相干结构，从根本上克服白噪声带来的高频噪点；
   - **零初始化与结构平价**：速度输出层（`Conv2d`）采用零初始化（Zero-Init）。

3. **严格确定性回退契约与温度控制分离**：
   - **`deterministic_fallback`**：显式布尔开关，为 `True` 时完全跳过 ODE 积分直接返回确定性预测 $\mu_t$，提供严格的 $D_0$ 零误差结构平价承诺；
   - **`noise_scale`**：概率源扰动强度的连续温度旋钮（Temperature knob）。澄清：在训练后具有非零速度漂移的网络中，$\text{noise\_scale}=0.0$ 从 $x_0=0$ 沿学得速度面积分，产生均值漂移流线，不等于 $\mu_t$。两者语义严格解耦。

4. **高阶数值 ODE 积分求解器 (`ODESolver`)**：
   - 实现包含 `euler`（一阶）、`midpoint`（二阶）、`heun`（二阶）及 `rk4`（四阶 Runge-Kutta）的高精度并行数值积分器；
   - 支持动态可配置积分步数（如 $N \in [5, 20]$），平滑权衡推断速度与生成质量。

5. **端到端世界模型集成 (`LatentForecaster`) 与严格密码学治理**：
   - 支持单源多轨迹并行滚动（Single-Source Multi-Trajectory Rollout），通过批次 $K$ 倍扩展保证各采样轨迹独立的自回归历史缓冲；
   - 训练启动前执行 Fail-Closed 密码学一致性比对（比对 $D_0$ 的 `split_hash`、`normalizer_hash`、`seed` 与运行环境；比对残差统计量哈希）；
   - 在 `optimizer.step()` 前严格检测全量梯度范数有限性，遭遇非有限值立即熔断；
   - 目录备份与覆盖保护严格后置于 Preflight 成功之后。

## 影响 (Consequences)

### 正向收益 (Positive)
- **空间相干性突破**：从白噪声采样演进为基于连续流形演化的相干流场生成，杜绝非物理高频噪点；
- **尺度均衡与稳定收敛**：残差尺度归一化保证 64 个潜通道以同等权重受训，对齐高阶湍流能量分布；
- **严格科学定义与治理保障**：清晰对标 ArchesWeatherGen，规范 Latent CFM 术语，实现端到端 Fail-Closed 密码学防呆与非有限梯度熔断。

### 权衡与约束 (Trade-offs & Constraints)
- **推断计算开销**：单步采样需执行 $N$ 次速度网络评估（例如 10 步 Midpoint 积分），计算量高于单次高斯采样，但得益于潜空间尺寸极小（$16 \times 16 \times 64$），单步仍可在毫秒级完成。
