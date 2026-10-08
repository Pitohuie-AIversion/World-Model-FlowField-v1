# ADR-008: 世界模型核心软件契约第一阶段抽象架构 (World Model Core Contract Abstraction - Phase 1)

## 状态 (Status)
**Accepted**

## 决策背景 (Context & Problem Statement)
World-Model-FlowField-v1 经过前期研发，已完整建立了端到端流体物理世界模型体系：
- 2D shear_flow 数据链路与零泄漏划分；
- 空间潜流形编解码器 (Encoder2D / Decoder2D)；
- 确定性时空 Transformer 动力学推演 (LatentSTTransformer)；
- 纯潜空间历史推演缓冲区 (HistoryBuffer)；
- 课程学习推前自回归训练 (Curriculum / Pushforward)；
- 条件高斯概率潜动力学 (Conditional Gaussian Latent Dynamics)；
- 残差潜流匹配生成动力学 (Residual Latent Conditional Flow Matching, Latent CFM / FM-R2)；
- 连续偏微分方程 (Navier-Stokes & 标量输运方程) 残差物理闭环；
- 严格的密码学 Provenance 与 Fail-Closed 治理体系。

然而，现有代码中散落着对特定数据集的先验假设：
1. 状态默认被假定为 `[u, v, p, s]` 且通道数固定为 4；
2. 动力学模型与 conditioning 层直接暴露并强依赖 Reynolds 数 ($Re$) 和 Schmidt 数 ($Sc$)；
3. 训练脚本与下游评测直接感知具体动力学类型（deterministic、Gaussian、Flow Matching），调用方法名称（`forward`、`predict_distribution`、`sample_next_latent`）与签名不统一；
4. 数据流从 DataLoader 到 Trainer 的 batch 解析由各脚本自行处理，字段分散。

为了在**零行为改变 (Zero Behavior Changes)** 的前提下，支撑未来世界模型持续横向与纵向扩展：
\[
\text{shear\_flow} \to \text{Rayleigh-Bénard} \to \text{Cylinder/Obstacle} \to \text{Geometry/Boundary} \to \text{Sparse Observation} \to \text{Language Context} \to \text{Robot Action}
\]
亟需在底层确立第一层稳定的通用软件契约。

---

## 核心设计与契约决策 (Decision Drivers & Core Contracts)

本轮建立四个核心软件契约抽象（位于 `src/contracts/`）：

### 1. StateSpec (物理状态规格契约)
- **定位**：明确定义物理状态中包含哪些语义变量、对应通道索引及空间维度。
- **结构**：
  ```python
  @dataclass(frozen=True)
  class StateSpec:
      variables: Tuple[str, ...]
      num_channels: int
      spatial_dim: int = 2
  ```
- **shear_flow 规范**：`SHEAR_FLOW_STATE_SPEC = StateSpec(variables=("u", "v", "p", "s"), num_channels=4, spatial_dim=2)`。
- **特性**：不可变（frozen）、具备张量维度校验 (`validate_tensor`)、非法输入 Fail-Fast。

### 2. Context (多模态条件上下文契约)
- **定位**：将分散的物理参数、边界、几何、外部强迫、语言及控制输入统一收敛于单一上下文契约。
- **分层语义**：
  ```python
  @dataclass
  class Context:
      physical: Optional[PhysicalContext] = None  # 当前阶段启用: Re, Sc
      geometry: Optional[Any] = None             # 预留: SDF, Obstacle Mask
      boundary: Optional[Any] = None             # 预留: 周期/无滑移壁面算子
      forcing: Optional[Any] = None              # 预留: 时空外力场
      language: Optional[Any] = None             # 预留: 语言目标描述
      action: Optional[Any] = None               # 预留: 机器人/控制动作
  ```
- **兼容性与冲突防御**：提供 `from_re_sc` 与 `to_re_sc`，在固定输入下与历史 ConditioningMLP 数值 bitwise equal；同时通过 `resolve_context` 统一兼容旧参数。在同时提供 `Context` 与旧 `(re, sc)` 参数时，执行**严格闭门校验 (Fail-Closed)**：
  - **输入类型与转换前防御**：支持 Python `float`、`int`，PyTorch 实数 Tensor 及 NumPy 实数标量/数组与合法形状序列；在可能发生信息丢失的类型转换前，统一执行原始输入校验，递归深度拒绝布尔配置（`bool`，防止被当作 0.0/1.0）与复数参数（`complex`，防止隐式丢弃虚部）；
  - **整数精确表示界限**：支持的整数输入范围严格限定在 IEEE 754 binary64 精确无损表示界 $[-(2^{53}), 2^{53}]$ 内；整型张量安全提升至 int64、NumPy 数组提取极值标量转 Python int 进行双边上下界比较，彻底消除 abs 溢出与窄整型比较截断漏洞；序列逐标量执行递归校验；超出界限明确抛出 `ValueError` 拒绝；
  - **无损精度比较**：双侧输入独立规范化为 float64 CPU 比较副本（显式绑定 `device="cpu"`），禁止向 Context 既有 dtype 降精度或整型截断，确保比较结果完全对称且对输入顺序不敏感；
  - **校验与计算分离**：校验副本仅用于冲突检查，通过后模型计算路径严格沿用原张量与数据流，不发生就地修改；
  - **非空与有限性防御**：显式要求输入非空 (`numel() > 0`)，显式调用 `torch.isfinite` 拒绝 NaN 与 Inf；
  - **形状契约与一致性容差**：明确标量 `()` / `(1,)` 与批次向量 `(B,)` / `(B, 1)` 边界，严禁标量与批次张量之间隐式广播匹配；一致性容差设定为绝对容差 `atol=1e-6`（为人为规定的工程一致性绝对容差，用于吸收 float32/float64 尾数表示微弱差异，非数学意义上的严格相等）；超出容差或类型/范围/形状/有限性不符立即抛出异常，杜绝静默覆盖。

### 3. WorldModelBatch (统一世界模型 Batch 契约)
- **定位**：连接 DataLoader 与 Trainer/Evaluator 的通用 Batch 容器。
- **结构 (Phase 1.1 单一事实源治理)**：
  ```
  WorldModelBatch
  ├── history: Tensor (B, L, C, Ny, Nx)
  ├── state_spec: StateSpec (显式必填，禁止静默默认 shear_flow)
  ├── future: Optional[Tensor] (B, H, C, Ny, Nx)
  ├── context: Optional[Context]
  ├── coordinates: Optional[Dict[str, Any]] (dt, time)
  └── metadata: Dict[str, Any] (source_file, traj_idx, start_t, cluster_id)
  ```
- **单一事实源 (Single Source of Truth)**：`boundary` 与 `geometry` 权威数据仅存于 `Context`。`WorldModelBatch.boundary` 与 `WorldModelBatch.geometry` 仅作为只读 `@property` 代理至 `context`，杜绝双重事实状态。
- **领域适配器 (Domain Adapters)**：通用契约不再预设剪切流。提供 `shear_flow_batch_adapter` 与 `collate_shear_flow_batch` 负责注入 `SHEAR_FLOW_STATE_SPEC` 与周期边界条件。
- **兼容性**：实现 `collections.abc.Mapping` 协议，旧代码以 `batch["history"]` 或 `batch["re"]` 访问 100% 透明兼容；提供 `.to(device)` 批量设备迁移能力。

### 4. LatentDynamics (统一潜动力学接口契约)
- **定位**：统一确定性、高斯概率以及流匹配动力学的前向与采样契约。
- **核心能力**：
  - `predict_mean(latent_history, context=...) -> Tensor`: 预测期望/确定性下一时刻潜状态；
  - `sample(latent_history, context=..., num_samples=1, **kwargs) -> Tensor`: 采样生成下一时刻潜状态；
  - `rollout(latent_history, context=..., horizon=H, ...)`: 统一潜空间多步自回归推演。
- **三分支架构实现**：
  ```
  LatentDynamics
  ├── DeterministicLatentDynamics (包装 LatentSTTransformer)
  ├── GaussianLatentDynamics (包装 LatentSTTransformer + VarianceHead2D)
  └── FlowMatchingLatentDynamics (包装 D0 Backbone + LatentFlowMatcher)
  ```
- **检查点加载安全审计**：`LatentFlowMatcher._load_from_state_dict` 保持 `strict=True` 语义，对 `residual_scale` 张量的形状、通道数、有限性（无 NaN/Inf）与严格正性进行前置验证，非法或畸变权重直接记录至 `error_msgs` 触发严格失败，绝不进行静默修复。

---

## 全局调用拓扑 (Architecture Dataflow Topology)

```
StateSpec (Explicit Schema)
    │
    ▼
WorldModelBatch [ history, state_spec, future, context, coordinates, metadata ]
    │
    ▼ (Representation Subspace: Encoder2D)
Latent History Z_{t-L+1:t}
    │
    ▼
LatentDynamics ◄──── Context (physical: Re, Sc; geometry, boundary, forcing, language, action)
    │
    ├── predict_mean() -> Z_{t+1} (D0 Mean)
    └── sample()       -> Z_{t+1} (Gaussian / Flow Matching ODE)
    │
    ▼ (Autoregressive rollout via HistoryBuffer)
Future Latent Sequence Z_{t+1:t+H}
    │
    ▼ (Representation Subspace: Decoder2D)
Decoded Future Physical Fields q_{t+1:t+H}
```

---

## 零行为改变保障 (Zero-Behavior-Change Invariant)

1. **数值等价性**：在相同权重与随机种子下，旧接口与新契约在单步预测及多步推演（$H=1, 4, 8$）上结果 bitwise equal；
2. **检查点兼容性**：Checkpoint schema 保持完全不变；D0、G1 及 FM checkpoint 仍能以 `strict=True` 正常加载；
3. **治理完整性**：Provenance 哈希校验、Fail-closed 拦截逻辑及非有限值截断保持严格生效；
4. **向后兼容性**：旧调用方式（`re=re, sc=sc`）与旧训练入口保持完全可用，无破坏性改动。
