# ADR-003: 产物分层治理规范与符号链接向后兼容策略

- **状态 (Status)**: Accepted
- **日期 (Date)**: 2026-09-24
- **决策人 (Deciders)**: 软件架构与持续集成工程团队
- **相关文档**: [outputs/figures/README.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/figures/README.md), [outputs/logs/README.md](file:///root/mzy/Flow%20Field%20Prediction%20in%20World%20Models/World-Model-FlowField-v1/outputs/logs/README.md)

---

## 1. 背景与上下文 (Context)

随着实验多阶段推进（Stage A 基线、Stage B 表征、Closure-R4 物理损失消融、Horizon-R1 跨度消融、Horizon-R2 DDP 扩展），`outputs/` 目录积累了数十个指标 JSON、上百张图表以及 20+ 个训练日志：
1. **扁平堆叠导致检索混乱**：图表、日志散落在根目录下，开发人员难以分清具体实验阶段与成果归属；
2. **自动化测试与老旧脚本依赖硬编码路径**：早期测试与分析脚本（如 `tests/test_training_convergence.py`、`scripts/analyze_training_convergence.py`）硬编码通过 `outputs/train_*.log` 或 `outputs/figures/*.png` 查找文件；
3. **版本控制策略冲突**：Git 仓库需要将轻量元数据、LaTeX 报告和核心指标纳入追踪，同时严格屏蔽大型权重与日志文本。

## 2. 架构决策 (Decision)

我们确立了**产物分层治理与符号链接向后兼容架构（Layered Artifacts with Symlink Back-Compat）**：

1. **子目录分层归档标准**：
   - **图表库**：`outputs/figures/` 划分为 `manuscript/`（论文正文图）、`closure_r4/`（物理消融曲线）、`horizon_r1/`（跨度消融）、`benchmark/`（基准对比）与 `qualitative/`（定性流场云图）；
   - **日志库**：`outputs/logs/` 划分为 `closure_r4/`、`horizon_r1/`、`horizon_r2/` 与 `early_ablation/`；
   - **指标库**：`outputs/metrics/` 统一存储标准化的指标 JSON，孤立临时文件予以归档清理；
   - **表格库**：`outputs/tables/` 存储论文汇报专用 LaTeX 源码；
   - **划分库**：`outputs/splits/` 存储哈希校验的数据划分契约。

2. **同名相对符号链接向后兼容 (Symlink Preservation Policy)**：
   - 所有移动至分层子目录的文件，均在原有父路径下建立同名相对符号链接（如 `outputs/train_closure_r4_*.log -> logs/closure_r4/train_closure_r4_*.log`）；
   - 依赖 `Path.glob()` 或固定路径的历史测试和旧分析脚本无需做任何破坏性修改，即插即用保证 100% 测试通过率。

3. **版本控制两级白名单策略**：
   - 根目录 `.gitignore` 与 `outputs/.gitignore` 统一规则：默认忽略所有非结构产物（`*.pt`, `*.ckpt`, `*.log`, `*.png` 等）；
   - 显式白名单豁免：`!tables/*.tex`, `!manifests/*.json`, `!metrics/*.json`, `!splits/*.json`, `!normalization/*.pt`, `!figures/**/*.png`, `!*/README.md`。

## 3. 架构影响与权衡 (Consequences)

### 正向收益 (Positive)
- **代码空间井然有序**：目录层次分明，每个实验批次自成闭环；
- **零破坏性回归**：自动化测试（155 个用例）全部绿色通过，未修改一行无关测试逻辑；
- **版本控制干净纯粹**：有效杜绝了几十兆临时日志与权重误提交至 Git 仓库的风险。

### 负面代价与应对 (Negative & Mitigations)
- 相对符号链接在 Windows 或跨文件系统打包时可能存在限制；在 Linux/Docker 运行时环境下具有原生最佳支持，且未来新增脚本优先使用规范化子目录路径。
