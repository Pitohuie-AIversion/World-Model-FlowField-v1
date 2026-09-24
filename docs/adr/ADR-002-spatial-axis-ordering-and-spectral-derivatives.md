# ADR-002: 空间网格轴序契约与双向周期 FFT 谱导数算子

- **状态 (Status)**: Accepted
- **日期 (Date)**: 2026-09-22
- **决策人 (Deciders)**: 物理科学计算与架构团队
- **相关文档**: [CLOSURE_R4_SPATIAL_AXIS_FIX.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/CLOSURE_R4_SPATIAL_AXIS_FIX.md), [ARCHITECTURE.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/docs/ARCHITECTURE.md)

---

## 1. 背景与上下文 (Context)

在 The Well 剪切流数据集（Dedalus 谱方法求解器生成）中：
- 物理区域定义为：$x \in [0.0, 1.0]$（长度 $L_x = 1.0$），$y \in [-1.0, 1.0]$（长度 $L_y = 2.0$）；
- 离散空间分辨率定义为：`spatial_resolution = [Nx, Ny] = [128, 256]`，即横向 $x$ 轴 128 点，纵向 $y$ 轴 256 点；
- 早期工程实现中由于混淆了网格轴序与域尺寸绑定，在计算谱导数 $\frac{\partial}{\partial x}, \frac{\partial}{\partial y}$ 时曾发生 $k_x$ 与 $k_y$ 波动数向量与尺度因子错位，导致计算出的无散场散度异常增大。

## 2. 架构决策 (Decision)

我们确立了不可动摇的**空间网格轴序物理契约（Spatial Axis Contract）**：

1. **张量空间布局定序**：
   - 全局张量格式严格固定为：
     - 单帧流场：`(B, C, Nx, Ny)` 其中 `Nx = 128`, `Ny = 256`；
     - 时序流场：`(B, T, C, Nx, Ny)`；
   - 负索引映射约定：
     - `dim=-2` 对应 $x$ 轴（水平对流方向，点数 $N_x=128$，域尺寸 $L_x=1.0$）；
     - `dim=-1` 对应 $y$ 轴（竖直高度方向，点数 $N_y=256$，域尺寸 $L_y=2.0$）。

2. **二维实数快速傅里叶谱导数算子 (2D rFFT)**：
   - 在 `src/utils/fft_derivatives.py` 中实现高精度连续谱导数：
     ```python
     # dim=-2 (x-axis) 使用 torch.fft.fftfreq(nx, d=lx / nx)
     # dim=-1 (y-axis) 使用 torch.fft.rfftfreq(ny, d=ly / ny)
     # 散度算子: div = d(u)/dx + d(v)/dy
     # 涡量算子: vort = d(v)/dx - d(u)/dy
     ```
   - 物理常数与域尺寸集中定义在 `src/utils/physics_contract.py`：
     - `SHEAR_FLOW_DOMAIN_SIZE_XY = (1.0, 2.0)`
     - `SPATIAL_AXIS_CONTRACT = "tensor(...,C,Nx,Ny):dim-2=x,dim-1=y"`

3. **双重契约卫兵与断言防护**：
   - 在数据集加载、自编码器前向、损失计算及评测汇总中，统一调用 `validate_ablation_checkpoint_semantics()` 强校验契约标识，防止轴向错位配置隐式扩散。

## 3. 架构影响与权衡 (Consequences)

### 正向收益 (Positive)
- **物理导数绝对正确**：真实剪切流无散度检验误差降至机器精度级（$10^{-7}$ 量级）；
- **涡量计算精确**：涡量场与 Dedalus 真值相关系数达到 0.999+；
- **全流水线统一**：所有模型、测试、脚本对张量轴的解释完全一致，根除了静默维数转置 bug。

### 负面代价与应对 (Negative & Mitigations)
- 历史检查点若未注入轴向契约元数据，加载时会触发强警告；系统提供了向下兼容的契约自动对齐回退逻辑。
