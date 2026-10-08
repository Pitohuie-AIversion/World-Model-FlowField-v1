# StocBench 随机流场数据审计与世界模型数据契约兼容性评估报告

## 一、审计概述与结论

- **审计结论**：**PASS WITH CONDITIONS**
  - **核心依据**：
    1. 真实官方数据文件 `traj_seed_42.npy` 与 `step_seed_100.npz` 均成功下载、校验并实测通过；
    2. 本地计算的 SHA-256 与远程 Hugging Face Git LFS 声明完全一致（字节级吻合）；
    3. 数据维度、dtype、物理采样时间步长及求解器保存缩放语义已全部核实，无 NaN/Inf 或退化常数；
    4. 同一初态下 5000 个未来样本的数值分叉得到严格证实（空间标准差均值 0.4620，与保存的 `mean`/`std` 偏差为 0.0）；
    5. 单通道涡量数据成功通过现有 `StateSpec` 与 `WorldModelBatch` 最小契约；
    6. 新增定向测试及既有相关模块回归测试全部通过。
  - **保留条件（CONDITIONS）**：
    1. **数据许可未明**：上游代码仓库标注 MIT 许可证，但 Hugging Face 数据集卡片未声明明确的 Dataset License，在正式分发或商业化前需向上游团队确认；
    2. **表示模型通道不匹配**：当前仓库既有 Encoder/Decoder 权重均为 4 通道（$u, v, p, s$），不能直接复用于单通道涡量数据，后续需训练单通道自编码器或采用物理空间模型。

---

## 二、真实数据来源与版本追溯

| 标识项 | 对应内容 | 证据来源 |
| :--- | :--- | :--- |
| **本项目评估 Commit** | `40e5a619096ede506c8b933619781113fca3e158` | 本地 `git rev-parse HEAD` |
| **StocBench 代码仓库** | `https://github.com/tum-pbs/stocbench` | 官方开源代码 |
| **StocBench 代码 Revision** | `2e519f153738972fb41c7c996ada8e1a4f4cb7a9` | GitHub 远程 `HEAD` |
| **StocBench 代码 License** | `MIT License` (Copyright (c) 2025 Sebastian Pfister) | `LICENSE` 文件 |
| **Hugging Face 数据集** | `https://huggingface.co/datasets/pfistse/stocbench-data` | 官方数据仓库 |
| **Hugging Face 数据 Revision** | `3a5f50398cf6d14f108190ace63a9beed5fbddf7` | API `repo_info.sha` |
| **数据集 License 状态** | 未显式声明（Card Data 为空） | Hugging Face 仓库元数据 |
| **论文引用** | arXiv:2608.22309 | *StocBench: A Benchmark for Generative Modeling of Stochastic Dynamics* |

---

## 三、下载文件完整性审计

下载遵循最小范围原则，仅获取指定的 1 个训练轨迹与 1 个单步分叉评测文件：

| 文件名称 | 相对路径 | 字节数 (Bytes) | 本地实测 SHA-256 | Git LFS 声明 SHA-256 | 一致性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `traj_seed_42.npy` | `data/stocbench/incns_stoc/64/traj_seed_42.npy` | `1,638,400,128` (~1.526 GiB) | `5a29c5620c0c30334877344d3a65166ca4c2a5a8e997df5e37d6c54c4a0d4c19` | `5a29c5620c0c30334877344d3a65166ca4c2a5a8e997df5e37d6c54c4a0d4c19` | 完全匹配 |
| `step_seed_100.npz` | `data/stocbench/incns_stoc/64/step_seed_100.npz` | `75,923,069` (~72.41 MiB) | `5d869c53b39ac81ca91d5dd12dafeb3315ddbbd407081865256368fe9cf25990` | `5d869c53b39ac81ca91d5dd12dafeb3315ddbbd407081865256368fe9cf25990` | 完全匹配 |

- **下载机制**：采用支持多线程与安全断点续传的 `aria2c`（结合系统环境既有代理），使用 `.download` 临时文件机制，在完成字节数与 SHA-256 双重校验后原子移动至最终目录。

---

## 四、真实数据结构与数值审计

### 1. 训练轨迹文件 (`traj_seed_42.npy`)

- **读取方式**：`np.load(..., mmap_mode="r", allow_pickle=False)` 内存映射安全读取。
- **数组维度**：`(500, 200, 1, 64, 64)`
- **数据类型**：`float32`
- **各轴语义**：
  1. 轴 0（`N=500`）：独立仿真轨迹数（500 条轨迹）；
  2. 轴 1（`T=200`）：时间快照数（200 帧）；
  3. 轴 2（`C=1`）：状态通道数（单通道二维涡量 $\omega$）；
  4. 轴 3（`Ny=64`）：垂直空间网格分辨率；
  5. 轴 4（`Nx=64`）：水平空间网格分辨率。
- **时间信息区分**：
  - **求解器内部推进步长**：$\Delta t_{\text{solver}} = 0.0001$（源自 `solver/configs/incns_stoc.yaml`）；
  - **相邻保存帧时间间隔**：$\Delta t_{\text{sample}} = 0.5$（求解器内部每隔 5000 步采样输出一帧）；
  - **物理时间跨度**：单条轨迹总时长 $T_{\text{phys}} = 200 \times 0.5 = 100.0$。
- **数值与分布特征**：
  - NaN 数量：0；Inf 数量：0；
  - 实测取值范围（抽样 5 条轨迹）：$\min = -5.3271$，$\max = 6.0329$，$\text{mean} = 0.0000$，$\text{std} = 0.9776$；
  - 变化性：未发现常数退化或全零帧。
- **缩放与标准化语义**：
  - 官方求解器在输出保存时执行了：$w_{\text{stored}} = (w_{\text{physical}} - \text{mean}) / \text{std}$，其中配置定义 $\text{mean} = 0.0, \text{std} = 3.0$（见 `solver/base.py` 行 151-153）；
  - **结论**：文件中存储的数值已经是经过 $3.0$ 倍除法缩放的无量纲数值，其标准差约为 $1.0$；真实物理涡量满足 $\omega_{\text{phys}} = 3.0 \times \omega_{\text{stored}}$。因此后续使用时**不得直接套用未经说明的双重标准化**。

### 2. 条件分叉评测文件 (`step_seed_100.npz`)

- **读取方式**：`np.load(..., allow_pickle=False)` 受控字典读取。
- **包含键名与形状**：
  - `init`: `(1, 64, 64)`，`float32`，给定的单帧初始高分辨率状态经 block-mean 下采样后的初态；
  - `raw`: `(5000, 1, 1, 64, 64)`，`float32`，在完全相同的 `init` 条件下，因未观测随机强迫演化一步产生的 5000 个真实可能未来；
  - `mean`: `(1, 64, 64)`，`float32`，官方预先计算的 5000 个未来的样本均值；
  - `std`: `(1, 64, 64)`，`float32`，官方预先计算的 5000 个未来的样本标准差（采用总体标准差 `ddof=0`）。
- **统计一致性校验**：
  - 对 5000 个样本沿成员轴重新计算：
    $$\text{diff}_{\text{mean}} = \max |\text{raw.mean}(0) - \text{mean}| = 0.000000\times 10^{0}$$
    $$\text{diff}_{\text{std}} = \max |\text{raw.std}(0, \text{ddof}=0) - \text{std}| = 0.000000\times 10^{0}$$
  - **结论**：实测统计值与保存值完全精确重合。

---

## 五、固定初态条件分叉的三层证据与边界

针对“固定初态的参考集合是否包含不同的真实下一状态”这一关键问题，我们建立了三层证据链并明确了其证据边界：

### 1. 三层证据体系

```
[Layer A: 生成机制]
官方求解器复制相同初态 init -> 注入不同伪随机数种子的傅里叶谱随机强迫 -> 仿真推进 sample_dt

[Layer B: 文件结构]
step_seed_100.npz 结构明确包含 1 个 init 与 5000 个对应的 raw 候选未来（K_ref = 5000）

[Layer C: 文件数值实测]
5000 个未来之间存在显著空间分布差异：
- 成员轴空间标准差：mean = 0.4620, min = 0.3895, max = 0.7850
- 任意两个分支的最大局部差值：|branch_0 - branch_1|_max = 2.6223
```

### 2. 证据边界与非推断声明

- **证据确证范围**：数据确实来源于真实的随机 Navier-Stokes 系统，在相同的单帧当前条件 $\omega_t$ 下，下一时刻流场 $\omega_{t+1}$ 确实呈现连续且具有明显方差的非确定性条件分布。
- **严谨边界（不成立的假设）**：
  1. 文件中仅保存了单步（1 步）的分叉集合，**未保存跨多步的长轨迹分叉树**；
  2. 样本之间存在显著差异，**不能直接证明条件分布是多峰分布**（可能仍接近多维高斯或偏态分布）；
  3. 不能由此推断“Flow Matching 必定优于高斯模型”；
  4. 不能由此推断“现有世界模型已经具有正确的概率校准”。这些结论需要后续严谨的模型训练与能量距离（Energy Distance）评测给出。

---

## 六、世界模型数据契约兼容性验证

### 1. 契约设计

- **`StateSpec` 配置**：
  ```python
  STOCBENCH_STATE_SPEC = StateSpec(
      variables=("vorticity",),
      num_channels=1,
      spatial_dim=2,
  )
  ```
- **批次张量对齐**：
  - 单帧历史条件 `history`：`(B, 1, 1, 64, 64)`；
  - 单步训练目标 `future`：`(B, 1, 1, 64, 64)`；
  - 分叉评测参考集合 `reference_ensemble`：`(B, K_ref, 1, 1, 64, 64)`。
- **结构隔离原则**：
  - 评测参考集合的成员轴 $K_{\text{ref}} = 5000$ 保持为独立评测张量，**不与时间轴 $H$ 混淆**，也**不强行塞入只支持单轨迹的 `WorldModelBatch.future`**。
  - 数据集不伪造压力 $p$ 与示踪剂 $s$，不补零凑 4 通道，不伪造 Reynolds/Schmidt 数。

### 2. 实测兼容性结果

- `WorldModelBatch` 成功实例化，包含：
  - `history.shape == (1, 1, 1, 64, 64)`
  - `future.shape == (1, 1, 1, 64, 64)`
  - `coordinates["dt"] == 0.5`
  - `context.boundary == "periodic"`
  - `metadata`: `{"dataset_id": "stocbench", "dataset_revision": "3a5f50398cf6d14f...", "source_file": "traj_seed_42.npy", ...}`
- `.to("cpu")` 与 `.to("cuda:0")` 迁移测试完全通过；
- 继承的 Mapping 协议访问（`batch["history"]`, `batch["future"]`, `batch["dt"]`）完全兼容。

### 3. 数据集与用途隔离

- `traj_seed_42.npy`：仅作为训练数据源（`StocBenchTrainDataset`，可切分 99,500 个滑动窗口）；
- `step_seed_100.npz`：仅用于数据审计及后续概率生成评估（`StocBenchReferenceEnsemble`）；
- 单元测试明确断言：尝试将 `.npz` 作为训练轨迹加载会触发 fail-fast 异常。

---

## 七、新增组件与修改清单

1. `configs/data/stocbench.yaml`：StocBench 数据源、物理参数、校验哈希及本地路径配置；
2. `src/data/stocbench_dataset.py`：
   - `STOCBENCH_STATE_SPEC`：单通道涡量状态规范；
   - `StocBenchTrainDataset`：内存映射训练数据集；
   - `StocBenchReferenceEnsemble`：分叉参考集合解析器与统计校验器；
   - `stocbench_batch_adapter`：数据契约适配器；
3. `scripts/prepare_stocbench_data.py`：支持 `preflight` 与 `download-audit` 双阶段的准备与审计工具；
4. `tests/test_stocbench_data_audit.py`：包含 11 个覆盖合法读取、非法输入校验、统计一致性、退化防御及真实数据校验的完整测试集；
5. `docs/data/STOCBENCH_DATA_AUDIT.md`：本技术审计报告。

---

## 八、进入下一阶段的前置条件评估

| 前置条件项 | 状态 | 评估说明 |
| :--- | :---: | :--- |
| **真实数据可获取性** | **满足** | 已完成远端预检与本地全量下载校验 |
| **数据物理与时间语义** | **满足** | $\Delta t_{\text{sample}} = 0.5$，已区分内部求解步长与缩放系数 |
| **条件随机分叉证据** | **满足** | 5000 个未来样本方差明显，统计完全自洽 |
| **最小数据契约兼容** | **满足** | 单通道 `StateSpec` 与 `WorldModelBatch` 验证通过 |
| **模型算法保持不变** | **满足** | 未修改 Transformer、Gaussian、Flow Matching 任何核心算法 |
| **测试与代码回归** | **满足** | 11 个定向测试全部通过，102 个既有模块测试全部通过 |
| **表示模型匹配（待办）** | *需下阶段解决* | 既有自编码器为 4 通道，后续需要针对 1 通道涡量适配自编码器或探索直接流匹配 |

---

## 九、审计产物与执行证据归档索引

本次规范化审计运行完整记录在独立运行目录：

`outputs/data_audit/stocbench/run_stocbench_audit_v1/`

| 产物文件 | 说明 | 关键索引项 |
| :--- | :--- | :--- |
| `manifest.json` | 完整元数据清单与配置快照 | 包含环境版本、远程 commit、本地哈希及下载参数 |
| `audit_results.json` | 数值审计与结构尺寸量化结果 | 包含两个文件的实测统计、分叉指标与契约验证状态 |
| `execution.log` | 数据获取与审计全流程控制台日志 | 记录预检、下载阶段耗时与数据流关键动作 |
| `pytest.log` | 定向测试套件完整执行原始日志 | 记录 11 个测试用例（含真实文件集成测试）的全部输出 |
| `summary.md` | 本次运行的简要技术提炼 | 包含通过状态、核心指标与关键结论 |
