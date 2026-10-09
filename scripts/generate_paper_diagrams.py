#!/usr/bin/env python3
"""Generate publication-grade conceptual and architectural diagrams for the First-Author Synthesis.

Diagram 1: Data Contracts and Relationship Pipeline
  - Clarifies: DNS solver generation, Initial conditions, Boundary conditions,
    Model historical inputs (L=4), and Future supervision targets (H steps).

Diagram 2: World Model Architecture and Latent State Rollout
  - Clarifies: Spatial Autoencoder (64x), AdaLN-Zero physical modulation,
    Spatial-Temporal Factorized Attention, FIFO latent buffer rollout,
    and FFT spectral loss supervision.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = Path("outputs/figures/paper_synthesis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "figure.dpi": 300,
    })


def create_data_relationship_diagram():
    fig, ax = plt.subplots(figsize=(15, 7.5), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_dns = "#e2e8f0"      # Slate grey
    c_bc = "#fef3c7"       # Amber
    c_in = "#dbeafe"       # Soft Blue
    c_tgt = "#fee2e2"      # Soft Red
    c_model = "#e0e7ff"    # Indigo

    # 1. DNS Box (Left / Top)
    rect_dns = patches.FancyBboxPatch((4, 62), 26, 32, boxstyle="round,pad=1.2,rounding_size=2",
                                     linewidth=1.5, edgecolor="#64748b", facecolor=c_dns)
    ax.add_patch(rect_dns)
    ax.text(17, 90, "1. 离线数据生成 (DNS Solver 独有)", fontsize=11, fontweight="bold", ha="center", color="#1e293b")
    ax.text(17, 82, "• 高精度拟谱法/微步长连续推进\n• 压力泊松方程逆解 (Pressure Poisson)\n• 初始随机摄动生成\n• 时间积分步长 dt 采样",
            fontsize=9, ha="center", va="center", color="#334155")
    ax.text(17, 66, "⚠️ 仅用于离线仿真，训练/推理完全不可见", fontsize=8.5, fontweight="bold", ha="center", color="#b91c1c")

    # 2. Boundary Condition Box (Top Right)
    rect_bc = patches.FancyBboxPatch((35, 62), 60, 32, boxstyle="round,pad=1.2,rounding_size=2",
                                    linewidth=1.5, edgecolor="#d97706", facecolor=c_bc)
    ax.add_patch(rect_bc)
    ax.text(65, 90, "2. 结构硬约束 (Boundary Conditions - 双周期环状拓扑)", fontsize=11, fontweight="bold", ha="center", color="#92400e")
    ax.text(65, 80, "流向 (x, Lx=1.0) 与 法向 (y, Ly=2.0) 双向周期性:  q(x + Lx, y) = q(x, y),  q(x, y + Ly) = q(x, y)",
            fontsize=9.5, fontweight="bold", ha="center", color="#78350f")
    ax.text(65, 69, "• 空间卷积层: 统一硬编码 padding_mode='circular'\n• 空间微分算子: 采用连续二维正交复数复傅里叶谱基 (FFT)，拓扑边界天然自动满足",
            fontsize=9, ha="center", color="#451a03")

    # 3. Model Dynamic Input (Bottom Left)
    rect_in = patches.FancyBboxPatch((4, 8), 42, 45, boxstyle="round,pad=1.2,rounding_size=2",
                                    linewidth=1.8, edgecolor="#2563eb", facecolor=c_in)
    ax.add_patch(rect_in)
    ax.text(25, 48, "3. 模型动态输入 (Model Inputs)", fontsize=12, fontweight="bold", ha="center", color="#1e40af")
    ax.text(25, 41, "历史观测时序滑窗 (严格锁定 L = 4 帧):", fontsize=9.5, fontweight="bold", ha="center", color="#1e3a8a")
    ax.text(25, 33, "q_{t-3:t} ∈ R^(B × 4 × 4 × 128 × 256)\n4 物理通道: [u, v, p, s]\n(水平速度 u, 垂直速度 v, 压力 p, 示踪物 s)",
            fontsize=9, ha="center", color="#1e293b", family="monospace")
    ax.text(25, 21, "无量纲环境物理条件:\nRe ∈ [10^3, 10^5] (雷诺数) | Sc ∈ [0.1, 1.0] (施密特数)\n由 PhysicalContext 编码调制",
            fontsize=8.5, ha="center", color="#0f172a")
    ax.text(25, 12, "注: 初始条件 (t=0) 仅作为滑窗起点锚点，无特殊输入特权", fontsize=8.5, ha="center", style="italic", color="#475569")

    # 4. Supervision Target Box (Bottom Right)
    rect_tgt = patches.FancyBboxPatch((53, 8), 42, 45, boxstyle="round,pad=1.2,rounding_size=2",
                                     linewidth=1.8, edgecolor="#dc2626", facecolor=c_tgt)
    ax.add_patch(rect_tgt)
    ax.text(74, 48, "4. 损失监督目标 (Supervision Targets)", fontsize=12, fontweight="bold", ha="center", color="#991b1b")
    ax.text(74, 41, "未来真实演化场 (Future Ground Truth):", fontsize=9.5, fontweight="bold", ha="center", color="#7f1d1d")
    ax.text(74, 33, "q*_{t+1:t+H} ∈ R^(B × H × 4 × 128 × 256)\n推演视界 H ∈ [1, 30] 步",
            fontsize=9, ha="center", color="#1e293b", family="monospace")
    ax.text(74, 23, "严密防穿越契约 (No Information Leakage):\n自回归推演时模型严禁读取未来帧！\n第 k 步自回归输入仅由模型前序自生成潜状态提供",
            fontsize=8.5, fontweight="bold", ha="center", color="#991b1b")
    ax.text(74, 12, "仅用于反向传播阶段计算: 场损失 + 谱散度损失 + 谱涡量损失", fontsize=8.5, ha="center", color="#450a0a")

    # Connective Arrows
    # DNS -> Input & Target
    ax.annotate("", xy=(25, 54), xytext=(17, 61),
                arrowprops=dict(arrowstyle="->", lw=2, color="#64748b", ls="--"))
    ax.annotate("", xy=(74, 54), xytext=(22, 61),
                arrowprops=dict(arrowstyle="->", lw=2, color="#64748b", ls="--"))

    # Boundary -> Inputs & Targets
    ax.annotate("", xy=(30, 54), xytext=(55, 61),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#d97706"))
    ax.annotate("", xy=(70, 54), xytext=(65, 61),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#d97706"))

    # Inputs -> Targets comparison indicator
    ax.annotate("自回归推演生成 vs 真值对齐", xy=(52, 28), xytext=(47, 28),
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#7c3aed"),
                fontsize=9, fontweight="bold", ha="center", va="bottom", color="#6d28d9")

    fig.suptitle("图 1：剪切流世界模型数据生成、输入流、边界先验与监督目标逻辑关系契约图",
                 fontsize=14, fontweight="bold", y=0.98)

    out_file = OUT_DIR / "fig1_data_relationship_and_contracts.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated {out_file}")


def create_model_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 8), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Box styles
    b_enc = dict(boxstyle="round,pad=0.8", fc="#eff6ff", ec="#3b82f6", lw=1.5)
    b_dyn = dict(boxstyle="round,pad=0.8", fc="#f5f3ff", ec="#8b5cf6", lw=1.8)
    b_dec = dict(boxstyle="round,pad=0.8", fc="#ecfdf5", ec="#10b981", lw=1.5)
    b_loss = dict(boxstyle="round,pad=0.8", fc="#fff1f2", ec="#f43f5e", lw=1.5)

    # 1. Inputs
    ax.text(8, 75, "历史流场输入 q_{t-3:t}\n(B, 4, 4, 128, 256)\n4 通道 [u,v,p,s]",
            ha="center", va="center", bbox=dict(boxstyle="square,pad=0.6", fc="#f8fafc", ec="#94a3b8", lw=1.2),
            fontsize=9, fontweight="bold")

    ax.text(8, 25, "物理上下文\nRe ∈ [10^3, 10^5]\nSc ∈ [0.1, 1.0]",
            ha="center", va="center", bbox=dict(boxstyle="square,pad=0.6", fc="#fefce8", ec="#eab308", lw=1.2),
            fontsize=9, fontweight="bold")

    # 2. Encoder
    ax.text(26, 75, "空间编码器 (Encoder2D)\n3 级周期卷积 (stride=2)\n8× 空间下采样\n通道 4 → 64",
            ha="center", va="center", bbox=b_enc, fontsize=9.5, fontweight="bold")

    ax.text(42, 75, "潜空间轨迹 Z_{t-3:t}\n(B, 4, 64, 16, 32)\n64× 空间体积压缩",
            ha="center", va="center", bbox=dict(boxstyle="square,pad=0.6", fc="#e0e7ff", ec="#6366f1", lw=1.2),
            fontsize=9, fontweight="bold")

    # 3. AdaLN-Zero
    ax.text(26, 25, "物理调制网络\nPhysicalContext MLP\n生成调制参数 (γ, β)",
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", fc="#fef08a", ec="#ca8a04", lw=1.2),
            fontsize=9)

    # 4. Latent ST-Transformer
    ax.text(60, 50, "纯潜空间动力学推进 (LatentSTTransformer)\n"
                    "• 时空因子化注意力 (Spatial-Temporal Factorized Attention)\n"
                    "• AdaLN-Zero 条件调制注入\n"
                    "• 状态转移推演: Z_{t+1} = T_θ(Z_{t-3:t}, Re, Sc)\n"
                    "• FIFO 潜状态缓冲机制: 弹出 Z_{t-3}，推入 Z_{t+1}\n"
                    "• 自回归内部自由滚动 H 步 (完全无需网格循环)",
            ha="center", va="center", bbox=b_dyn, fontsize=10, fontweight="bold")

    # 5. Decoder
    ax.text(82, 75, "空间解码器 (Decoder2D)\n3 级周期转置卷积\n8× 空间上采样\n+ 零均值压力投影 ∫p=0",
            ha="center", va="center", bbox=b_dec, fontsize=9.5, fontweight="bold")

    ax.text(94, 75, "预测物理场 q_{t+1}\n(B, 4, 128, 256)",
            ha="center", va="center", bbox=dict(boxstyle="square,pad=0.6", fc="#d1fae5", ec="#059669", lw=1.2),
            fontsize=9, fontweight="bold")

    # 6. Loss Supervision
    ax.text(78, 20, "多目标物理损失闭包 (FFT 连续谱微分)\n"
                    "L_total = L_field + 0.01·L_div + 0.05·L_vort + L_enst + L_tracer\n"
                    "• L_div: 谱散度连续性自由约束 (FFT: ik_x u + ik_y v)\n"
                    "• L_vort: 谱空间高阶涡量守恒 (FFT: ik_x v - ik_y u)\n"
                    "• L_enst: 拟涡能守恒与示踪物有界全变差惩罚",
            ha="center", va="center", bbox=b_loss, fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle="->", lw=2, color="#334155")
    ax.annotate("", xy=(18, 75), xytext=(15, 75), arrowprops=arrow_props)
    ax.annotate("", xy=(35, 75), xytext=(34, 75), arrowprops=arrow_props)
    ax.annotate("", xy=(49, 65), xytext=(45, 70), arrowprops=arrow_props)
    ax.annotate("", xy=(49, 40), xytext=(35, 28), arrowprops=dict(arrowstyle="->", lw=2, color="#ca8a04"))
    ax.annotate("", xy=(73, 75), xytext=(70, 60), arrowprops=arrow_props)
    ax.annotate("", xy=(89, 75), xytext=(88, 75), arrowprops=arrow_props)

    # Internal Loop Arrow in Transformer (FIFO Rollout)
    ax.annotate("自回归潜循环 (FIFO Rollout H 步)", xy=(55, 62), xytext=(65, 62),
                arrowprops=dict(arrowstyle="->", lw=2, color="#7c3aed", connectionstyle="arc3,rad=-0.5"),
                fontsize=9, fontweight="bold", color="#6d28d9", ha="center")

    # Decoder -> Loss & Future Target -> Loss
    ax.annotate("", xy=(82, 32), xytext=(82, 65), arrowprops=dict(arrowstyle="->", lw=1.8, color="#e11d48", ls="--"))

    fig.suptitle("图 2：流场世界模型整体架构、时空因子化注意力、FIFO 纯潜滚动与连续谱物理损失闭环",
                 fontsize=14, fontweight="bold", y=0.98)

    out_file = OUT_DIR / "fig2_world_model_architecture_and_fifo.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Generated {out_file}")


if __name__ == "__main__":
    setup_style()
    create_data_relationship_diagram()
    create_model_architecture_diagram()
