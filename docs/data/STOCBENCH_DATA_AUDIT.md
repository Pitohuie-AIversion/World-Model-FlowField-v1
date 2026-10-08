# StocBench 随机流场数据审计与世界模型数据契约兼容性评估报告

## 一、审计概述与结论

- **当前结论**：**BLOCKED——数据下载、数值审计及最小契约适配已有执行记录；数据使用许可仍待确认，正式训练准入暂不关闭。**
  - **已确认的工程事实（通过项）**：
    1. 真实官方数据文件 `traj_seed_42.npy` 与 `step_seed_100.npz` 均成功下载、校验并实测通过；
    2. 本地计算的 SHA-256 与远程 Hugging Face Git LFS 声明完全一致（字节级吻合）；
    3. 数据维度、dtype、物理采样步长及求解器保存缩放语义已全部核实，无 NaN/Inf 或退化常数；
    4. 同一初态下 5000 个未来样本的数值离散度得到实测检验（空间标准差均值 0.4620，与保存的 `mean`/`std` 偏差为 0.0）；
    5. 单通道涡量数据成功通过现有 `StateSpec` 与 `WorldModelBatch` 最小契约；
    6. 普通回归与离线测试完全解耦，干净检出环境可完全离线运行（10 passed, 1 skipped），真实数据验收执行 Fail-Closed 机制（无数据时断言报错终止，有数据时 11 passed）。
  - **阻塞项与准入边界（BLOCKED 根因）**：
    1. **数据使用许可未明确**：上游代码仓库标注 MIT 许可证，但 Hugging Face 数据集卡片未声明明确的 Dataset License（Card Data 为空，无 License tag）。按照严格工程治理标准，未获明确许可前**不得进入正式模型训练与对外公开发布**，因此训练准入暂不关闭；
    2. **单通道表示模型尚未建立**：当前仓库既有权重为 4 通道 shear_flow 自编码器，不能直接复用于单通道涡量数据，后续需开展单通道自编码器表示验证。

---

## 二、真实数据来源与版本追溯

| 标识项 | 对应内容 | 证据来源 |
| :--- | :--- | :--- |
| **本项目评估 Commit** | `40e5a619096ede506c8b933619781113fca3e158` (后接入提交 `a0e35c5`) | 本地 `git rev-parse HEAD` |
| **StocBench 代码仓库** | `https://github.com/tum-pbs/stocbench` | 官方开源代码 |
| **StocBench 代码 Revision** | `2e519f153738972fb41c7c996ada8e1a4f4cb7a9` | GitHub 远程 `HEAD` |
| **StocBench 代码 License** | `MIT License` (Copyright (c) 2025 Sebastian Pfister) | `LICENSE` 文件 |
| **Hugging Face 数据集** | `https://huggingface.co/datasets/pfistse/stocbench-data` | 官方数据仓库 |
| **Hugging Face 数据 Revision** | `3a5f50398cf6d14f108190ace63a9beed5fbddf7` | API `repo_info.sha` |
| **数据集 License 状态** | **未显式声明**（Card Data 为空，未打 license tag） | Hugging Face 仓库元数据 |
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
  4. 轴 3（`H=64`）：空间网格第一轴（未转置，对应求解器首轴模态 $k_x$，即物理空间 $x$ 方向）；
  5. 轴 4（`W=64`）：空间网格第二轴（未转置，对应求解器次轴模态 $k_y$，即物理空间 $y$ 方向）。
  - **空间坐标轴映射约定**：上游求解器在谱空间中第一空间轴为 $k_x$、第二空间轴为 $k_y$。适配器维持未转置（Non-transposed）直接读取，张量位置 $H$ 对应物理 $x$ 坐标，$W$ 对应物理 $y$ 坐标。在 $[0, 2\pi)$ 方形周期域上网格等距（$\Delta x = \Delta y = 2\pi/64$），后续计算方向相关物理算子时，第一空间轴算子为 $\partial_x$，第二空间轴算子为 $\partial_y$。
- **时间与物理时机区分**：
  - **求解器内部推进步长**：$\Delta t_{\text{solver}} = 0.0001$；
  - **相邻保存帧时间间隔**：$\Delta t_{\text{sample}} = 0.5$（求解器内部每隔 5000 步采样输出一帧）；
  - **保存时机与首末跨度**：官方求解器完成 warmup 之后，推进 5000 步（即 $t=0.5$）保存第 0 帧；推进至第 $200 \times 5000$ 步保存第 199 帧（$t=100.0$）。
    - 首末保存帧之间的时间跨度为：$199 \times 0.5 = 99.5$；
    - 求解器累计运行时长为：$200 \times 0.5 = 100.0$。
- **数值与分布特征**：
  - NaN 数量：0；Inf 数量：0；
  - 实测取值范围（抽样 5 条轨迹）：$\min = -5.3271$，$\max = 6.0329$，$\text{mean} = 0.0000$，$\text{std} = 0.9776$；
  - 变化性：未发现常数退化或全零帧。
- **缩放语义（存储尺度 vs 求解器尺度）**：
  - 官方求解器在输出保存时执行了无量纲缩放：$w_{\text{stored}} = (w_{\text{solver}} - 0.0) / 3.0$（见 `solver/base.py` 行 151-153）；
  - **结论**：文件中存储数值的标准差约为 $1.0$；乘以 $3.0$ 恢复的是求解器内部无量纲尺度，求解器本身基于无量纲参数（$\nu=0.001, L=2\pi$）运行，不能草率认定为有具体物理单位的真实量纲物理量。后续模型使用时**禁止套用未经说明的双重标准化**。

### 2. 条件分叉评测文件 (`step_seed_100.npz`)

- **读取方式**：`np.load(..., allow_pickle=False)` 受控字典读取。
- **包含键名与形状**：
  - `init`: `(1, 64, 64)`，`float32`，给定的单帧初始状态经 block-mean 下采样后的初态；
  - `raw`: `(5000, 1, 1, 64, 64)`，`float32`，在相同 `init` 条件下，因未观测随机强迫演化一步产生的 5000 个真实候选未来；
  - `mean`: `(1, 64, 64)`，`float32`，官方预存的 5000 个未来的样本均值；
  - `std`: `(1, 64, 64)`，`float32`，官方预存的 5000 个未来的总体标准差（`ddof=0`）。
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
[Layer A: 生成机制与源码追溯]
审查 StocBench 官方求解器 solver/create_incns_stoc_dataset.py 确认：
求解器复制同一初态 init，注入不同伪随机数种子的谱空间随机强迫推进 sample_dt

[Layer B: 文件结构与完整性互证]
文件结构将一个初态 (init) 与多个未来样本 (raw) 关联；结合固定版本的生成代码和文件完整性校验，支持这些样本来自同一初态的判断。
（注：单个初态数组加多个未来数组，单靠这种静态文件结构本身并不能证明其动态生成过程，必须与 Layer A 的源码及 Git SHA 形成证据闭环）

[Layer C: 文件数值实测与离散性验证]
5000 个未来样本在空间各点展现出非零方差与显著离散度：
- 成员轴空间标准差：mean = 0.4620, min = 0.3895, max = 0.7850
- 任意两个分支的最大局部差值：|branch_0 - branch_1|_max = 2.6223
（注：非零方差证明的是样本存在离散性；随机性来源和独立生成机制需结合求解代码与连续性实测共同定性）
```

### 2. 证据边界与非推断声明

- **证据确证范围**：数据确实来源于具有谱空间随机外力的二维 Navier-Stokes 仿真系统；在相同的初态 $\omega_t$ 下，下一时刻流场 $\omega_{t+1}$ 确实呈现连续且具有明显方差的非确定性条件分布。
- **严谨边界（不成立的假设）**：
  1. 文件中仅保存了单步（1 步）的分叉集合，**未保存跨多步的长轨迹分叉树**；
  2. 样本之间存在显著离散度，**不能直接证明条件分布是多峰分布**（可能仍接近单峰偏态分布）；
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

- **标准 Batch 构造示例（严格遵守冻结契约）**：
  ```python
  from src.contracts.context import Context, PhysicalContext
  from src.contracts.batch import WorldModelBatch

  # 1. 边界仅存储在 Context.boundary；固定求解器参数放入 PhysicalContext.extra
  context = Context(
      physical=PhysicalContext(
          extra={"nu": 0.001, "drag": -0.1, "epsilon": 1.0}
      ),
      boundary="periodic",
  )

  # 2. Batch 构造不使用独立 boundary 字段，通过 context.boundary 访问
  batch = WorldModelBatch(
      history=history,                   # (B, 1, 1, 64, 64)
      future=future,                     # (B, 1, 1, 64, 64)
      state_spec=STOCBENCH_STATE_SPEC,   # num_channels=1
      context=context,
      coordinates={"dt": 0.5},
      metadata=metadata,
  )
  assert batch.boundary == "periodic"
  assert batch.context.to_re_sc() == (None, None)
  ```
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

## 七、测试执行策略与环境边界实测

为确保代码库在无外部大文件/无 GPU 依赖的环境下保持纯净可维护，同时真实数据验收具备严格 Fail-Closed 特性，本工程建立明确解耦的双执行范围，并在本地环境完成全覆盖实测验证：

### 1. 双执行范围设计与实测对比

| 执行范围 | 依赖条件与环境变量 | 执行命令 | 实测结果 | 行为判定 |
| :--- | :--- | :--- | :--- | :--- |
| **定向离线单元测试（模拟缺失大文件目录）** | 模拟指定无大文件目录环境 (`STOCBENCH_REQUIRE_REAL_DATA=0`) | `STOCBENCH_DATA_DIR=/tmp/clean_empty_checkout python -m pytest -q tests/test_stocbench_data_audit.py` | `...........s` <br> **`11 passed, 1 skipped in 2.80s`** | **PASS**（前 11 项小型合成数据及防覆写测试 100% 通过；第 11 项因缺失真实数据显式 skip，不阻断常规 CI 流程） |
| **真实数据验收测试（缺失数据场景）** | 强制执行真实数据验收 (`STOCBENCH_REQUIRE_REAL_DATA=1`)，但目标文件缺失 | `STOCBENCH_DATA_DIR=/tmp/clean_empty_checkout STOCBENCH_REQUIRE_REAL_DATA=1 python -m pytest -q tests/test_stocbench_data_audit.py` | `...........F` <br> **`AssertionError: FAIL-CLOSED: Real trajectory file ... not found!`** (退出码 1) | **FAIL-CLOSED**（明确触发断言异常，绝不静默放行） |
| **真实数据验收测试（数据就绪场景）** | 本地真实 1.6 GiB 数据落地，强制验收 (`STOCBENCH_REQUIRE_REAL_DATA=1`) | `STOCBENCH_REQUIRE_REAL_DATA=1 python -m pytest -q tests/test_stocbench_data_audit.py` | **`12 passed in 4.12s`** | **PASS**（12 项测试全绿通过，包括滑动窗口切分、5000 候选未来统计对齐与防覆写校验） |
| **相关世界模型模块回归** | 核心契约、自编码器、条件高斯、潜空间流匹配 | `python -m pytest -q tests/test_core_contracts.py tests/test_dataset.py tests/test_splits.py tests/test_encoder_decoder.py tests/test_probabilistic_interfaces.py tests/test_latent_flow_matching.py` | **`102 passed in 9.00s`** | **PASS**（零破坏性改动） |

> **边界声明（测试范围收窄）**：上述“无数据环境测试”仅证明了定向测试脚本在数据缺失时具备跳过大文件验收的离线降级能力，**不代表本环境已证明“一台没有任何历史缓存与依赖的全新物理机器”能够通过全量测试**。

### 2. 警告信息（Warnings）根因分析与原始日志精准定位

在仓库全量回归测试（`python -m pytest -q`）中输出的 **`541 passed, 14 warnings`**，其全部 14 个 warnings 来源已被原始日志精确定位：
- **唯一来源文件**：`tests/test_compile_roundtrip.py: 14 warnings`；
- **告警类别**：`DeprecationWarning`；
- **原始告警堆栈**：
  ```text
  tests/test_compile_roundtrip.py: 14 warnings
    /root/miniconda3/envs/seagent/lib/python3.12/site-packages/torch/jit/_script.py:362: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
      warnings.warn(
  ```
- **性质定性**：来自已有的模型 TorchScript 编译回环测试，属于 PyTorch 2.10 废弃 `torch.jit.script_method` 的上游 API 迁移提示；
- **与本轮关系**：本轮新增的 StocBench 数据适配与审计模块（`src/data/stocbench_dataset.py`、`tests/test_stocbench_data_audit.py`）产生 **0 个 warning**。

---

## 八、审计产物与执行证据归档索引

本次规范化审计运行完整记录在独立运行目录：

`outputs/data_audit/stocbench/run_stocbench_audit_v1/`

### 1. 产物清单与关键索引项

| 产物文件 | 说明 | 关键索引项 |
| :--- | :--- | :--- |
| `manifest.json` | 完整元数据清单与配置快照 | 包含环境版本、远程 commit、本地哈希及下载参数 |
| `audit_results.json` | 数值审计与结构尺寸量化结果 | 包含两个文件的实测统计、分叉指标与契约验证状态 |
| `execution.log` | 数据获取与审计全流程控制台日志 | 记录预检、下载阶段耗时与数据流关键动作 |
| `pytest.log` | 定向测试套件完整执行原始日志 | 记录测试用例（含真实文件集成测试）的全部输出 |
| `summary.md` | 本次运行的简要技术提炼 | 包含通过状态、核心指标与关键结论 |

### 2. 防覆写机制实测证据（持久化归档与并发安全保证）

为防止重复执行审计时静默覆盖历史证据，`scripts/prepare_stocbench_data.py` 实施了原子排他分配与并发安全保护：
- **原子排他分配机制**：通过 `allocate_exclusive_audit_dir` 使用操作系统原子系统调用 `Path.mkdir(exist_ok=False)`。若多个进程/线程并发申请同一 `run_id`，利用操作系统内核的排他性捕获 `FileExistsError`，自动追加时间戳与递增序列号重试，彻底杜绝 check-then-act 竞态条件；
- **实测生成的两套持久化证据目录（已永久保存在磁盘，供随时查验）**：
  1. `outputs/data_audit/stocbench/run_anti_overwrite_verified/manifest.json`
  2. `outputs/data_audit/stocbench/run_anti_overwrite_verified_20261008_180405/manifest.json`
- **并发自动化测试保障**：在 `tests/test_stocbench_data_audit.py` 的测试用例 `test_audit_output_anti_collision_and_overwrite_protection` 中，使用 10 线程并发竞争压测，确证 10 个并发工作者均获得互斥独立目录，且基线证据未被修改。

---

## 九、后续科研推进建议

本轮完成的是“**StocBench 数据准备、审计与单通道最小前向链路验证**”，而非“概率流场世界模型完成”。进入下一轮前，建议紧紧围绕世界模型的**潜空间动力学主线**推进：

1. **核心下一步**：单通道涡量 Encoder/Decoder 2D 表示模型训练与验证（本轮不直接加载四通道自编码器权重，使用独立单通道实例建立新基线）；
2. **核心回答问题**：潜空间压缩与重建会不会把原始随机未来之间的差异抹平？
3. **隔离保护原则**：`step_seed_100.npz` 继续严格封存为概率分叉评测基准，不得参与自编码器的训练或超参数调优。

---

## 十、代码状态与版本追踪审计（交付闭环披露）

### 1. 新增测试文件 `tests/test_stocbench_model_smoke.py` 纳入版本控制

- **定位**：单通道最小链路与接口冒烟测试套件（共 6 项用例）；
- **纳入版本库**：正式纳入 Git 索引并提交（Commit `9f4d82b`），杜绝工作区存在未跟踪核心测试文件的问题；
- **测试覆盖与实测状态**：
  1. `test_single_channel_autoencoder_smoke`：单通道自编码器前向及有限性（PASSED）；
  2. `test_decoder_pressure_projection_isolated_for_single_channel`：单通道下保持 `project_pressure=False` 隔离压力规范处理（PASSED）；
  3. `test_latent_dynamics_forward_default_context_smoke`：Deterministic / Gaussian / Flow Matching 三种动力学在 `PhysicalContext.extra` 缺省物理条件下的前向与采样（PASSED）；
  4. `test_stocbench_reference_ensemble_dimension_and_latent_projection`：分叉参考集合 $K_{\text{ref}}=5000$ 维度契约与潜空间投影（PASSED）；
  5. `test_stocbench_physical_scaling_roundtrip`：$\times 3.0$ 存储到物理尺度无损往返缩放（PASSED）；
  6. `test_train_and_reference_data_isolation`：训练集（`traj_seed_42.npy`）与评测集（`step_seed_100.npz`）的物理隔离与防泄漏（PASSED）。
- **早期草稿报错根因回溯（澄清非模型缺陷）**：
  前期草稿运行出现的两处报错已被追溯并准确定性：
  - `TypeError: LatentSTTransformer.__init__() got unexpected keyword argument 'in_channels'`：系草稿传参误写为 `in_channels`，正确参数名为 `latent_channels`；
  - `TypeError: VarianceHead2D.__init__() got unexpected keyword argument 'in_channels'`：系草稿传参误写为 `in_channels`，正确签名要求 `embed_dim, latent_channels`；
  - 结论：世界模型核心组件天然支持单通道潜空间，早期报错纯属测试草稿调用方式不匹配，而非模型架构缺陷，无需重构核心网络。

### 2. `src/models/decoder.py` 与冻结基线保持零差异（零代码变动）

- **核对事实**：冻结版本 `40e5a61` 中的 `Decoder2D` 构造参数已包含 `out_channels`（默认 4）与 `project_pressure`（默认 `False`），天然支持传入 `out_channels=1, project_pressure=False`；
- **决策结论**：收回任何非必要修改，**完全保持 `40e5a61` 源码不变**；
- **证据核对**：`git diff 40e5a619096ede506c8b933619781113fca3e158 HEAD -- src/models/decoder.py` 输出完全为空，既有 4 通道剪切流模型、动态覆盖测试及压力零均值规范行为 100% 保持完全一致。

### 3. 工作区状态与隔离边界声明

- **本轮交付物状态**：本轮新增与适配的目标代码、测试及报告已全部提交；
- **工作区隔离范围**：工作区仍保留其他任务历史修改（`scripts/export_report_pdf.py`，用于调整导出 PDF 超时时间）以及历史报告/图表文件。测试在包含该隔离修改的工作区上执行，经核查对数据加载与世界模型逻辑无任何副作用。
