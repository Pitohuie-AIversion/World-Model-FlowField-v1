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
[ 输入历史流场 q_{t-L+1:t} ] (B, L, 4, 128, 64)
            │
    ┌───────┴───────┐
    ▼               ▼
[ Encoder2D ]   [ ConditioningMLP ] (Re, Sc -> 对数嵌入 -> 128-dim)
(8x 卷积下采样)       │
    │               ▼
    └──────────► [ LatentSTTransformer (6层时空解耦注意力) ]
                    │
                    ▼  (AdaLN-Zero 调制 & 残差潜状态更新)
            [ HistoryBuffer (FIFO 潜空间自回归循环) ]
                    │
                    ▼ (推演 H 步潜状态序列 Z_{t+1:t+H})
            [ Decoder2D ] (转置卷积 8x 空间上采样)
                    │
                    ▼ (非就地正交投影规范)
            [ 压力零均值投影: p <- p - mean(p) ]
                    │
                    ▼
            [ 预测物理流场 q_{t+1:t+H} ] (B, H, 4, 128, 64)
                    │
    ┌───────────────┴───────────────┐
    ▼                               ▼
[ L_div 散度损失 ]              [ L_vort 涡量损失 ]
||div(u)||^2 = 0            ||curl(u) - omega||^2
(二维周期 FFT 谱梯度)         (二维周期 FFT 谱旋度)
    │                               │
    └───────────────┬───────────────┘
                    ▼
       [ 梯度穿透反向传播到 Transformer ]
```

---

## 2. 空间潜流形编解码器 (Encoder2D & Decoder2D)

### 2.1 结构规格与压缩率
- **输入维度**：$q = [u, v, p, s] \in \mathbb{R}^{B \times 4 \times 128 \times 64}$；
- **潜状态维度**：$Z \in \mathbb{R}^{B \times 64 \times 16 \times 8}$；
- **压缩比率**：空间分辨率 $128 \times 64 = 8192$ 降至 $16 \times 8 = 128$，特征点数缩减 **64 倍**；单帧参数压缩率达 $4 \times$。

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
- **空间自注意力（Spatial Attention）**：对每一个时间步 $t \in [1, L]$，在空间潜网格 $16 \times 8$（共 128 个 Token）之间计算全连接自注意力，捕捉大尺度涡对的相互卷吸与配对；
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

## 5. 周期快速傅里叶变换导数与双重物理守恒反传

为了克服纯数据驱动在长程推演中发散的缺陷，系统通过二维快速傅里叶变换（2D FFT）在物理场上构建了连续偏微分正则化损失：

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

---

## 6. 架构决策记录导航 (Architecture Decision Records, ADR)

系统核心设计决策已纳入轻量级 ADR 规范进行版本化治理：

- [ADR-001: 空间潜流形解耦与物理守恒梯度穿透架构](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-001-latent-world-model-architecture.md)
- [ADR-002: 空间网格轴序契约与双向周期 FFT 谱导数算子](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-002-spatial-axis-ordering-and-spectral-derivatives.md)
- [ADR-003: 产物分层治理规范与符号链接向后兼容策略](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/ADR-003-outputs-and-artifacts-governance.md)

详见 [docs/adr/README.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/adr/README.md)。
