# 服务器硬件与计算环境规范说明

> **归档状态**：已核验 (VERIFIED)  
> **核验日期**：2026-09-24  
> **平台环境**：AutoDL 西部三区（`westDC3` / `west-C`）容器实例  
> **容器主机名**：`autodl-container-5d66449c84-34c12a8a`  
> **系统内核**：Linux 5.15.0-112-generic x86_64 (Ubuntu 22.04.5 LTS)

---

## 1. 显卡与计算设备规格总览

通过底层 PCI 设备数据库查询、NVIDIA 驱动查询（`nvidia-smi -q`）及 PyTorch CUDA 运行时环境核验，当前计算节点配备 **2 张基于 NVIDIA Ada Lovelace 架构（AD103 核心，GeForce RTX 4080 规格）的 32GB 显存定制/虚拟化显卡**。

| 规格维度 | 具体参数与实测值 | 验证源与证据链 |
| :--- | :--- | :--- |
| **显卡数量** | **2 张**（GPU 0, GPU 1，支持双卡 DDP 分布式并行） | `torch.cuda.device_count() == 2` |
| **设备标识名称** | `NVIDIA vGPU-32GB` | `nvidia-smi -q` (`Product Name`) |
| **产品品牌系列** | `GeForce` | `nvidia-smi -q` (`Product Brand`) |
| **硬件芯片架构** | **NVIDIA Ada Lovelace** | `nvidia-smi -q` (`Product Architecture`) |
| **底层芯片型号** | **AD103**（标准桌面对应 GeForce RTX 4080） | PCI-SIG 官方设备数据库 (`pci.ids`) |
| **PCI Device ID** | `0x270410DE` (Vendor: `10DE`, Device: `2704`) | `nvidia-smi --query-gpu=pci.device_id` |
| **PCI Subsystem ID** | `0x179510DE` | `nvidia-smi -q` (`Sub System Id`) |
| **PCIe 总线地址** | GPU 0: `0000:99:00.0` / GPU 1: `0000:B1:00.0` | `nvidia-smi -q` (`Bus Id`) |
| **PCIe 带宽规格** | PCIe Gen4 (Link Width Max: 16x, Current: 8x) | `nvidia-smi -q` (`GPU Link Info`) |
| **显存总容量** | **32,760 MiB（约 31.47 GiB / 32 GB）/ 张** | `nvidia-smi -q` (`FB Memory Usage`) |
| **流多处理器 (SM)** | **76 个 SM**（对应 9,728 个 CUDA Core） | `torch.cuda.get_device_properties().multi_processor_count` |
| **算力等级** | **Compute Capability 8.9** | `torch.cuda.get_device_properties().major.minor` |
| **单卡功耗上限** | **320.00 W** (Default / Max Power Limit) | `nvidia-smi -q` (`Current Power Limit`) |
| **NVIDIA 驱动版本** | **580.76.05** | `nvidia-smi` |
| **CUDA 驱动版本** | **13.0** | `nvidia-smi` |

---

## 2. 硬件特性分析与平台定位

1. **核心芯片分析**：
   * 设备 ID `0x2704` 且拥有 76 个 SM、320W 标称功耗上限，与零售版 **GeForce RTX 4080 (AD103)** 核心完全同源。
2. **32GB 显存特性**：
   * 原生零售版 RTX 4080 显存为 16GB。本服务器识别为 32GB 显存，属于 AutoDL 等算力平台针对大模型与 SciML 科学计算需求定制/显存扩容的 **“RTX 4080 32G” 实例（或基于 Ada Lovelace 硬件切分的 vGPU-32GB 配置文件）**。
3. **计算能力与算子支持**：
   * 采用 Ada Lovelace 架构（CC 8.9），原生支持 FP8、BF16 混合精度（AMP）、第 4 代 Tensor Core，具备极高的高维张量收发与 FFT 频谱变换吞吐率。
   * 单卡 32GB 显存足以轻松支撑 $H=12$、$H=16$ 甚至 $H=32$ 潜空间多步展开，杜绝由于长推演导致的显存溢出风险。

---

## 3. 软件栈与运行时配置

| 组件 | 版本 | 说明 |
| :--- | :--- | :--- |
| **操作系统** | Ubuntu 22.04.5 LTS (x86_64) | 官方长期支持版 |
| **Linux 内核** | 5.15.0-112-generic | 容器共享宿主机内核 |
| **Python** | 3.10.x (`/root/miniconda3/envs/seagent/bin/python`) | 项目标准 Conda 环境 |
| **PyTorch** | 2.10.0+cu128 | 支持 CUDA 12.8+ 算子与分布式训练 |
| **分布式通信** | PyTorch DDP (`nccl` 后端) | 双卡分布式训练标准后端 |

---

## 4. 推荐训练配置契约

依据此双卡 32GB 硬件配置，本项目科学实验的标准推荐运行配置如下：

* **分布式并行**：`torch.distributed.run --nproc_per_node=2`（双卡 DDP）
* **批大小对齐**：$\text{Microbatch}=1$、$\text{Grad Accumulation}=4$、$\text{GPUs}=2 \implies B_{\text{eff}} = 8$
* **混合精度**：启用 `--use_amp`（自动利用 Ada Lovelace Tensor Core 加速并节省显存）
* **主端口**：`--master_port 29501`（避开默认端口冲突）

---

## 5. 快速核验复现命令

在终端中执行以下命令可随时核实服务器硬件与当前文档的吻合度：

```bash
# 1. 查询基础显存与卡数
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.device_id --format=csv

# 2. 查询详细架构、SM 与计算能力
python -c "
import torch
print('Count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name} | {p.total_memory / (1024**3):.2f} GB | SMs: {p.multi_processor_count} | CC: {p.major}.{p.minor}')
"
```
