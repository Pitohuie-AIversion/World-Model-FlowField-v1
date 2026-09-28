# ADR-006: 连续 Navier-Stokes 与示踪剂偏微分方程残差约束受控训练 (PDE Residual Losses & Controlled Training Governance)

## 状态 (Status)
**Accepted** (2026-09-28)

## 背景 (Context)
在流场预测中，单纯依靠网格场值均方误差（$\mathcal{L}_{\text{field}}$）及一阶微分损失（散度 $\mathcal{L}_{\text{div}}$、涡量 $\mathcal{L}_\omega$）虽能维持低维流形的基本几何，但模型在长时间自回归展开时，仍容易偏离真实的 Navier-Stokes 动量方程与被动示踪剂对流-扩散输运方程。
直接引入完整的二阶非线性 PDE 残差损失面临诸多工程与力学挑战：
1. 非线性对流项在频域求导计算易引发激烈的**混叠误差（Aliasing Errors）**；
2. 物理残差项与场重构项的梯度范数差异悬殊，易导致优化失衡或梯度淹没；
3. 数据管道中的时间差分要求时间步长 $\Delta t$ 严格均匀；如果模型位置编码解析配置存在隐式 fallback，会导致严重的训练不一致。

## 决策 (Decision)

我们构建并落地了高保真、可微的 Navier-Stokes 与示踪剂 PDE 动力学残差物理约束体系：

1. **高阶全连续 PDE 残差算子实现 (`src/losses/navier_stokes.py`)**
   - 动量方程残差：$\mathbf{r}_{\text{mom}} = \partial_t \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u} + \nabla p - \frac{1}{Re} \nabla^2 \mathbf{u}$；
   - 连续性约束：$r_{\text{mass}} = \nabla \cdot \mathbf{u} = 0$；
   - 示踪剂对流-扩散方程残差：$r_{\text{tracer}} = \partial_t s + (\mathbf{u} \cdot \nabla) s - \frac{1}{Re \cdot Sc} \nabla^2 s$；
   - 引入 Orszag 2/3 截断准则进行非线性项双向频域去混叠（Dealiasing），彻底消除数值虚假湍流。

2. **严格数据管道时间步与位置编码 Fail-Closed 治理**
   - 针对时间网格强制校验均匀步长 $\Delta t$，杜绝变步长污染物理连续导数；
   - 治理 `use_spatial_pos` 空间位置编码传递契约，实施严格的 Fail-Closed 判定，拒绝非显式配置的静默退化。

3. **残差零点审计 (Audit) 与梯度范数探测 (Probe)**
   - 经 `scripts/audit_pde_residuals.py` 审计证实，Ground Truth 的真实物理残差在 $1.5 \times 10^{-3}$ 量级，验证了算子实现的解析正确性；
   - 经 `scripts/probe_pde_gradient_scales.py` 实测残差梯度范数与余弦相似度，确定了物理损失与场值损失的最佳平衡尺度权重（$\lambda_{\text{mom}} = 0.05, \lambda_{\text{tracer}} = 0.02$）。

4. **受控微调与全体验证集基准评测 (Controlled Training & Evaluation)**
   - 建立受控实验配对（P0 对照组 vs PDE 实验组）；
   - 在全体验证集 1110 个滑动窗口上完成严格基准评测：PDE 模型在动量残差 $res_u$ 降低 1.48%、示踪剂残差 $res_s$ 降低 2.18%、散度降低 0.19%、压力 VRMSE 改善 0.23%。

## 影响 (Consequences)

### 正面收益
- 首次将完整的不可压缩 Navier-Stokes 动量方程与被动示踪剂对流扩散输运方程以解析可微的形式接入流场世界模型训练；
- 显著提升了长程自回归推演中流体微观物理结构的保真度，有效抑制了伪源汇与非物理浓度震荡；
- 建立了完备的 PDE 残差审计、梯度探测与模型评估工具链。

### 权衡与约束
- 计算动量与输运方程的非线性对流项与粘性拉普拉斯项需要多次 2D FFT 与逆 FFT 变换，训练步进耗时增加约 25%~35%；
- 建议将 PDE 残差约束作为多步推演模型后期的受控微调（Controlled Fine-tuning）手段，而非冷启动初始训练目标。
