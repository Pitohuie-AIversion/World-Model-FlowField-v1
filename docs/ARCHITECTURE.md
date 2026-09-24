# 流场世界模型 V1 系统架构与技术规范 (System Architecture)

> **项目名称**：World-Model-FlowField-v1  
> **基准物理场景**：The Well 2D 周期不可压缩剪切流（Periodic Incompressible Shear Flow with Passive Scalar）  
> **系统定位**：基于空间离散潜流形（Latent Manifold）与物理守恒正则化的流体动力学世界模型  

---

## 1. 总体架构拓扑 (Overall Architecture)

系统采用 **空间潜流形解耦架构（Latent World Model Architecture）**。将流体动力学求解分解为两个正交解耦的子空间：
1. **空间表示流形空间（Representation Subspace）**：负责将连续偏微分方程（PDE）的高维网格状态压缩映射到致密、光滑的低维离散物理潜状态；
2. **时序动力学推演空间（Dynamics Subspace）**：在低维潜空间中执行纯自回归时序前向演化，并由物理守恒损失（无散度与涡量一致性）穿透解码器雅可比矩阵提供全局约束。

```
[ 输入历史流场 q_{t-L+1:t} ] (B, L, 4, 128, 256)
            │
    ┌───────┴───────┐
    ▼               ▼
[ Encoder2D ]   [ ConditioningMLP ] (Re, Sc -> 对数嵌入 -> 128-dim)
(8x 卷积下采样)       │
    │               ▼
    └──────────► [ LatentSTTransformer (6层时空解耦注意力 + 2D PosEmb) ]
                    │
                    ▼  (AdaLN-Zero 调制 & 残差潜状态更新)
            [ HistoryBuffer (FIFO 潜空间自回归循环 & Pushforward) ]
                    │
                    ▼ (推演 H 步潜状态序列 Z_{t+1:t+H})
            [ Decoder2D ] (转置卷积 8x 空间上采样)
                    │
                    ▼ (非就地正交投影规范)
            [ 压力零均值投影: p <- p - mean(p) ]
                    │
                    ▼
            [ 预测物理流场 q_{t+1:t+H} ] (B, H, 4, 128, 256)
                    │
    ┌───────────────┼───────────────┐
    ▼                               ▼                               ▼
[ L_div 散度损失 ]              [ L_vort 涡量损失 ]              [ 可微 Leray 投影 ]
||div(u)||^2 = 0            ||curl(u) - omega||^2           P(u) = u - grad(inv_lap(div(u)))
(二维周期 FFT 谱梯度)         (二维周期 FFT 谱旋度)           (频域零散度投影)
    │                               │
    └───────────────┬───────────────┘
                    ▼
       [ 梯度穿透反向传播到 Transformer ]
```

---

## 2. 空间潜流形编解码器 (Encoder2D & Decoder2D)

### 2.1 结构规格与压缩率
- **输入维度**：$q = [u, v, p, s] \in \mathbb{R}^{B \times 4 \times 128 \times 256}$；
- **潜状态维度**：$Z \in \mathbb{R}^{B \times 64 \times 16 \times 32}$；
- **压缩比率**：空间分辨率 $128 \times 256 = 32768$ 降至 $16 \times 32 = 512$，特征点数缩减 **64 倍**；单帧参数压缩率达 $4 \times$。

### 2.2 物理双向周期卷积 (Circular Padding)
由于剪切流在水平（$x \in [0, 1]$）与竖直（$y \in [-1, 1]$）方向均服从周期性边界条件：
\[
q(x + L_x, y, t) = q(x, y, t), \quad q(x, y + L_y, t) = q(x, y, t)
\]
编解码器中所有 2D 卷积和转置卷积层均显式采用 `padding_mode="circular"`，彻底消除了常规零填充在边界处引发的非物理虚假剪切层与局部数值反射。

### 2.3 压力规范正交投影契约 (Zero-Mean Pressure Projection Policy)
Navier-Stokes 方程中不可压缩速度场仅取决于压力梯度 $\nabla p$，绝对压力具有物理规范自由度（Gauge Freedom $\int_\Omega p \, dx dy = 0$）。
在协议治理升级后（Protocol P1-2）：
1. **Decoder 保持纯净特征表达**：`Decoder2D` 采用未投影输出（`project_pressure=False`），避免在归一化特征空间强行去均值后因反归一化偏置重新引入均值漂移；
2. **反归一化物理空间正交投影**：在反归一化至真实物理空间后（或在物理评估阶段），严格施加非就地物理压力投影：
\[
p_{\text{proj}}(x, y) = p(x, y) - \frac{1}{|\Omega|}\iint_\Omega p(x', y') \, dx' dy'
\]
该设计同时满足了 Dedalus 谱方法求解器的零积分压力基准定义，又确保了特征空间梯度流动的完整性。

---

## 3. 动力学 Transformer 与自适应条件注入

### 3.1 时空因子化注意力 (Factorized Spatio-Temporal Attention)
为了在有限算力下建模全场大尺度涡旋拓扑并兼顾时序相关性，模型采用时空交替解耦注意力架构：
- **空间自注意力（Spatial Attention）**：对每一个时间步 $t \in [1, L]$，在空间潜网格 $16 \times 32$（共 512 个 Token）之间计算全连接自注意力，捕捉大尺度涡对的相互卷吸与配对；
- **时间自注意力（Temporal Attention）**：对潜空间中的每一个空间位置 $(i, j)$，在时间步序列 $t \in [1, L]$（4 帧历史）之间计算因果/双向注意力，捕捉对流输运的时间平移不变性。

### 3.2 物理参数 AdaLN-Zero 条件调制
物理控制参数 $c = [Re, Sc]$ 通过对数尺度变换映射：
\[
\tilde{c} = [\log_{10}(Re), \log_{10}(Sc)]
\]
经 `ConditioningMLP` 升维为 128 维物理特征向量，并注入到各层的自适应归一化层 `AdaptiveLayerNorm2D`：
\[
\hat{Z} = \gamma(c) \odot \text{LayerNorm}(Z) + \beta(c)
\]
其中调节权重 $\gamma, \beta$ 的输出投影头采用 **Zero-Init（零初始化）** 策略，保证在训练初期的动力学前向传递等价于标准恒等映射，避免了强物理条件扰乱注意力权重的冷启动。

### 3.3 2D 正弦余弦空间几何位置编码 (2D Sin-Cos Spatial Positional Embedding)
为弥补自注意力机制对连续空间几何拓扑位置信息的天然丢失，系统在潜特征序列输入注意力块前，显式注入确定性 2D 正弦余弦网格位置编码：
- **实现模块**：[src/models/positional_embedding.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/models/positional_embedding.py)（`get_2d_sincos_position_embedding`）；
- **数学映射**：在纵横两轴分别生成连续波长基频，正交拼接形成形状为 $(1, H_z \times W_z, C_z)$ 的位置张量；
- **归纳偏置**：赋予时空 Transformer 对剪切层横向剪切带与纵向对流位置的各向异性空间辨识力。

---

## 4. 纯潜空间闭环自由滚动机制 (HistoryBuffer)

为实现超长时序（30 步及以上）的高效推演，系统设计了纯潜空间自回归滑动缓冲区 `HistoryBuffer`：
1. **单次空间编码**：将历史 $L=4$ 帧物理场一次性投影至潜空间序列 $[Z_{t-3}, Z_{t-2}, Z_{t-1}, Z_t]$；
2. **纯潜自回归演化**：
   \[
   Z_{t+1} = Z_t + \Delta Z = Z_t + \text{Transformer}([Z_{t-3:t}], c)
   \]
   推演过程中通过先进先出（FIFO）机制循环更新潜状态，在推演全程不进行物理网格解码，计算开销与显存占用降低 90% 以上；
3. **终端一次性解码**：仅在生成全部未来 $H$ 步潜状态序列后，统一通过 Decoder 映射回高维物理场空间。

---

## 5. 周期快速傅里叶变换导数、Leray 投影与双重物理守恒反传

为了克服纯数据驱动在长程推演中发散的缺陷，系统通过二维快速傅里叶变换（2D FFT）在物理场上构建了连续偏微分正则化损失与频域投影：

### 5.1 周期谱导数算子 (FFT Spectral Derivatives)
利用傅里叶谱算子精确计算速度场的一阶空间导数与涡量：
\[
\frac{\partial u}{\partial x} = \mathcal{F}^{-1} \left( i k_x \mathcal{F}(u) \right), \quad \frac{\partial v}{\partial y} = \mathcal{F}^{-1} \left( i k_y \mathcal{F}(v) \right)
\]
- **速度散度**：$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}$
- **涡量场**：$\omega = \nabla \times \mathbf{u} = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$

### 5.2 物理守恒损失穿透与反向传播空间契约 (Protocol P1-1)
在训练推演阶段，将预测输出与真值送入联合损失函数：
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{field}} + \lambda_{\text{div}} \frac{1}{|\Omega|} \iint \|\nabla \cdot \mathbf{u}\|^2 dx dy + \lambda_{\omega} \frac{1}{|\Omega|} \iint \|\omega - \omega^*\|^2 dx dy
\]
- **$\mathcal{L}_{\text{field}}$ 优化空间**：默认在标准化特征空间（Normalized Space）计算，相当于按各通道方差的倒数 $1/\sigma_c^2$ 进行马氏加权，有效避免压力通道小方差（$\sim 10^{-4}$）导致的梯度淹没；
- **物理微分损失空间**：速度散度与涡量守恒约束具备明确物理量纲，**严格且始终在反归一化物理场上计算**；
- **梯度穿透**：由于 Decoder 内部完全采用标准可微卷积与激活层，即使在微调阶段冻结 Decoder 参数，物理梯度的雅可比链式法则仍能无损穿透至潜状态：
\[
\frac{\partial \mathcal{L}_{\text{div}}}{\partial Z} = \left( \frac{\partial q}{\partial Z} \right)^T \frac{\partial \mathcal{L}_{\text{div}}}{\partial q}
\]
实测证明，反传梯度的 $L_2$ 范数达到 61.65，驱动 Transformer 主动向“零散度、低拟能误差”的物理守恒流形对齐。

### 5.3 可微 Leray 谱投影算子 (Differentiable Leray Spectral Projection)
为了从架构底层无条件保障流场质量守恒，系统引入了端到端可微的二维 Leray 谱投影算子（`leray_projection_2d`，定义于 [src/utils/fft_derivatives.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/utils/fft_derivatives.py)）：
\[
\mathbb{P} \mathbf{u} = \mathbf{u} - \nabla \Delta^{-1} (\nabla \cdot \mathbf{u})
\]
在 2D 傅里叶频域中，对应的正交投影矩阵为：
\[
\widehat{\mathbb{P}}(\mathbf{k}) = \mathbf{I} - \frac{\mathbf{k} \mathbf{k}^T}{\|\mathbf{k}\|^2}, \quad \mathbf{k} \neq \mathbf{0}
\]
对于零频直流分量（$\mathbf{k}=\mathbf{0}$）保持平均流不变。该算子可直接插入在物理解码器之后，使输出速度场从数学上严格满足 $\nabla \cdot (\mathbb{P}\mathbf{u}) \equiv 0$。

---

## 6. 课程式长程推演与推前训练机制 (Curriculum Rollout & Pushforward Training)

针对长时间自回归自激累积误差引发的分布偏移（Distribution Shift）问题，系统在 [src/training/curriculum.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/training/curriculum.py) 与 [scripts/train_forecaster.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/scripts/train_forecaster.py) 中构建了推前训练与课程式展开机制：

1. **课程式滚动调度器 (CurriculumRolloutScheduler)**：
   - 支持 `doubling`（倍增递进：$1 \to 2 \to 4 \to 8 \to 16$）、`linear`（线性递增）与 `fixed` 三种策略；
   - 随 Epoch 演进平滑过渡推演窗口，先使模型学习精准短时局部梯度，再逐步施加长时动力学多步自回归约束；
2. **截断梯度推前预热机制 (Stop-Gradient Pushforward)**：
   - 训练期间在计算损失前，先在纯潜空间执行 $K$ 步自回归推演预热并截断历史梯度（`.detach()`）；
   - 使模型在偏离理想真值分布的“预测轨迹态”上继续前向一步并回传梯度，大幅提升自回归长程鲁棒性。

---

## 7. 高性能编译与检查点无缝兼容 (torch.compile & strip_compiled_prefix)

为了在训练中充分榨取现代 GPU 算力：
1. **PyTorch 2.x `torch.compile` 集成**：支持对 `LatentSTTransformer`、`Encoder2D` 与 `Decoder2D` 执行 Inductor 算子融合编译，吞吐量提升显著；
2. **前缀递归剥离保障严格加载**：针对编译后产生的 `_orig_mod.` 及子模块嵌套前缀，[src/utils/checkpoint.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/utils/checkpoint.py) 的 `strip_compiled_prefix` 会自动递归清理所有点分 `_orig_mod` 片段，确保已编译保存的权重可以在未编译的评测脚本中以 `strict=True` 100% 无缝载入。

## 8. 多尺度动能谱损失与湍流能级级联保持 (Multi-scale Energy Spectrum Loss & Energy Cascade Preservation)

针对深度神经网络在流体动力学自回归中普遍存在的“谱偏置（Spectral Bias）”现象——低频大尺度流动主导损失函数（>95% 能量占比），导致高频小尺度湍流涡结构过度平滑耗散或在高波数截断处发生虚假能量堆积与混叠（Aliasing/Shock），系统引入了多尺度径向壳层动能谱损失：

1. **可微批处理径向积分谱算子 (`compute_batched_radial_energy_spectrum`)**：
   - 在 [src/utils/fft_derivatives.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/utils/fft_derivatives.py) 中实现，利用二维实傅里叶变换（`rfft2`）计算速度场傅里叶能量密度，并通过 GPU 原生 `scatter_add_` 算子完成各向同性壳层积分；
   - 彻底避免 CPU 内存搬运与非微分循环，显存开销仅 260 KB，支持多维批次 `(B, T, C, Ny, Nx)` 且 100% 解析可导；
2. **多尺度对数动能谱损失模块 (`EnergySpectrumLoss`)**：
   - 在 [src/losses/spectral.py](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/src/losses/spectral.py) 中实现：
     \[
     \mathcal{L}_{\text{spec}} = \frac{1}{K} \sum_{k=1}^K w_k \left| \log_{10}(E_{\text{pred}}(k) + \epsilon) - \log_{10}(E_{\text{target}}(k) + \epsilon) \right|
     \]
   - **对数空间度量**：跨越多个能量数量级平衡不同尺度梯度的贡献权重；
   - **高频自适应强化**：支持通过系数 $\alpha$ 施加高波数强化加权 $w_k = 1 + \alpha \frac{k}{K_{\max}}$，主动抗击多步自回归模糊；
   - **全流程实时监测**：验证阶段同步输出低频（Low）、中频（Mid）与高频（High）的独立能谱误差，量化谱耗散演进。

---

## 9. 架构决策记录导航 (Architecture Decision Records, ADR)

系统核心设计决策已纳入轻量级 ADR 规范进行版本化治理：

- [ADR-001: 空间潜流形解耦与物理守恒梯度穿透架构](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-001-latent-world-model-architecture.md)
- [ADR-002: 空间网格轴序契约与双向周期 FFT 谱导数算子](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-002-spatial-axis-ordering-and-spectral-derivatives.md)
- [ADR-003: 产物分层治理规范与符号链接向后兼容策略](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-003-outputs-and-artifacts-governance.md)
- [ADR-004: 多尺度动能谱损失与课程自回归推前训练架构](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-004-multiscale-spectral-loss-and-curriculum-pushforward.md)

详见 [docs/adr/README.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/README.md)。

