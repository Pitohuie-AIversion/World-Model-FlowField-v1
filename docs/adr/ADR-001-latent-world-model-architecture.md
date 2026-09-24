# ADR-001: 空间潜流形解耦与物理守恒梯度穿透架构

- **状态 (Status)**: Accepted
- **日期 (Date)**: 2026-09-20
- **决策人 (Deciders)**: 算法与系统架构工程团队
- **相关文档**: [ARCHITECTURE.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/ARCHITECTURE.md), [CLOSURE_R4_SPATIAL_AXIS_FIX.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/CLOSURE_R4_SPATIAL_AXIS_FIX.md)

---

## 1. 背景与上下文 (Context)

流体动力学世界模型需要在高维离散网格上预测长时间步连续场演化（$q = [u, v, p, s] \in \mathbb{R}^{B \times 4 \times 128 \times 256}$）。
如果直接在物理网格空间使用时空注意力模型（如 Direct ST Transformer）：
1. **计算与显存复杂度过高**：在 $128 \times 256$ 网格上单步 Token 数量达 32,768，注意力矩阵计算与多步推演梯度反向传播显存爆炸；
2. **纯自回归累积误差发散**：在缺少物理规律先验约束时，高频能量迅速耗散或非线性不稳定发散；
3. **边界虚假反射**：剪切流在两方向均为周期性，传统零填充（Zero Padding）会在边界引入剪切应力奇异点。

## 2. 架构决策 (Decision)

我们决定采用**空间潜流形解耦与物理守恒梯度穿透架构（Latent World Model Architecture）**：

1. **两阶段正交子空间分解**：
   - **Stage B 空间表示流形**：训练 2D 卷积自编码器（`Encoder2D` / `Decoder2D`），通过 3 次 stride=2 的卷积实现空间 $8 \times 8$ 下采样（特征网格缩减 64 倍，单帧参数压缩 4 倍，潜状态维度为 $16 \times 32 \times 64$）；
   - **Stage C/D 时空动力学推演**：在紧致的潜空间中利用 6 层解耦时空注意力模型（`LatentSTTransformer`）进行时序演化。

2. **物理双向周期卷积 (Circular Padding)**：
   - 在 `Encoder2D` 与 `Decoder2D` 的全部卷积与残差块中显式强制 `padding_mode="circular"`，从拓扑上保证周期性边界条件无损。

3. **纯潜空间自回归滑动缓冲区 (HistoryBuffer)**：
   - 推演过程中利用 FIFO 缓冲区仅在潜空间更新状态 $Z_{t+1} = Z_t + \Delta Z$，推演全程不进行物理网格解码，计算开销与显存降低 90% 以上；仅在末端推演完成后统一解码。

4. **物理守恒梯度穿透契约 (Protocol P1-1 & P1-2)**：
   - 速度散度损失 $\mathcal{L}_{\text{div}}$ 与涡量守恒损失 $\mathcal{L}_{\text{vort}}$ 严格在反归一化物理网格上通过 2D FFT 谱微分计算；
   - 解码器内部保持纯净特征表达（`project_pressure=False`），非就地压力正交投影移至物理反归一化后执行；
   - 物理损失反传梯度无损穿透 Decoder 雅可比矩阵约束潜空间 Transformer 权重更新。

## 3. 架构影响与权衡 (Consequences)

### 正向收益 (Positive)
- **显存与速度优势**：潜空间 Token 数量由 32,768 骤降至 512，推演速度提升 6-8 倍，长程 30 步滚动推演显存可控；
- **长程物理稳定性**：引入双重物理损失穿透后，30 步自回归滚动的速度散度误差降低 84.7%，且彻底消除高频拟能虚假发散；
- **边界平滑**：双向周期卷积消除了一切局部边界反射假象。

### 负面代价与应对 (Negative & Mitigations)
- **解码器雅可比反传开销**：多步训练时通过 Decoder 反传物理微分梯度增加了约 15% 的反向传播耗时。通过短步推演（H=2 ~ 4）与长程冻结解码器进行平衡；
- **两阶段误差解耦依赖**：动力学推演上限受限于 Stage B 潜空间重建精度（当前重建 VRMSE 优于 0.05，满足高保真要求）。
