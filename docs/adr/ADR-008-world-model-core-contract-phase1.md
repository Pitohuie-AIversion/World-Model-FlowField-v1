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
- **兼容性**：提供 `from_re_sc` 与 `to_re_sc`，在固定输入下与历史 ConditioningMLP 数值 bitwise equal；同时通过 `resolve_context` 自动兼容旧的 `(re, sc)` 参数调用。

### 3. WorldModelBatch (统一世界模型 Batch 契约)
- **定位**：连接 DataLoader 与 Trainer/Evaluator 的通用 Batch 容器。
- **结构**：
  ```
  WorldModelBatch
  ├── history: Tensor (B, L, C, Ny, Nx)
  ├── future: Optional[Tensor] (B, H, C, Ny, Nx)
  ├── context: Optional[Context]
  ├── state_spec: StateSpec
  ├── coordinates: Optional[Dict[str, Any]] (dt, time)
  ├── boundary: Optional[Any] ("periodic")
  ├── geometry: Optional[Any] (None)
  └── metadata: Dict[str, Any] (source_file, traj_idx, start_t, cluster_id)
  ```
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

---

## 全局调用拓扑 (Architecture Dataflow Topology)

```
StateSpec
    │
    ▼
WorldModelBatch [ history, future, context, coordinates, metadata ]
    │
    ▼ (Representation Subspace: Encoder2D)
Latent History Z_{t-L+1:t}
    │
    ▼
LatentDynamics ◄──── Context (physical: Re, Sc; [future: geometry, boundary, forcing, language, action])
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
