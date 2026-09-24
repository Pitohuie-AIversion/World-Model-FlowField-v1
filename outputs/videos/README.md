# 剪切流时序视频与动画目录 (Outputs Videos)

本目录存储周期性剪切流（Periodic Shear Flow）DNS 真实物理轨迹及世界模型时序滚出的多媒体动态视频与元数据。

## 1. 核心视频资产

### 剪切流完整生命周期演化全景视频 (200 帧)
- **文件路径**: `outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames.mp4`
- **预览动图**: `outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames.gif`
- **元数据索引**: `outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames_metadata.json`
- **物理参数**: $Re = 10^4$, $Sc = 0.1$, 空间网格 $256 \times 512$, 物理域尺寸 $(L_x, L_y) = (1.0, 2.0)$
- **时序规格**: 200 帧完整时间轨迹 ($t = 0 \sim 199$), 20 FPS, 时长 10.0 秒, 分辨率 $2400 \times 1350$ (H.264 / yuv420p)

## 2. 画面布局与物理特征

视频采用 $2 \times 2$ 物理联动布局，所有面板使用固定统一的色标刻度，彻底杜绝帧间频闪：

1. **左上 (Panel 1) - 涡量场 $\omega = \partial_x v - \partial_y u$ (Colormap: `seismic`, $[-2.0, 2.0]$)**:
   - 清晰展示从界面扰动发展、开尔文-亥姆霍兹 (Kelvin-Helmholtz) 剪切层失稳卷起、双排反向对称大涡核的形成，到后期涡对吸引合并与细丝化耗散的完整拓扑结构演化。
2. **右上 (Panel 2) - 流向速度 $u$ (Colormap: `RdBu_r`, $[-0.45, 0.50]$)**:
   - 展示上下剪切层反向流动界面的强烈褶皱、折叠与对流传输。
3. **左下 (Panel 3) - 被动示踪剂浓度 $s$ (Colormap: `inferno`, $[-0.35, 0.45]$)**:
   - 展示流体物质界面的剧烈卷吸、缠绕混合与多尺度扩散。
4. **右下 (Panel 4) - 速度幅值 $|U| = \sqrt{u^2 + v^2}$ (Colormap: `viridis`, $[0.0, 0.50]$)**:
   - 捕捉动能集中区域及涡核高剪切能量边界的动态迁移动向。

## 3. 复现与自定义渲染命令

```bash
# 渲染完整 200 帧 2x2 四联全景视频与预览 GIF
python scripts/generate_shear_flow_video.py \
  --hdf5_path /root/autodl-tmp/datasets/shear_flow/data/valid/shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5 \
  --sim_idx 0 \
  --fps 20 \
  --layout quad \
  --output_video outputs/videos/shear_flow_dns_re1e4_sc0.1_sim0_quad_200frames.mp4

# 渲染 1x3 横排三联视频 (涡量 / 流速 u / 示踪剂)
python scripts/generate_shear_flow_video.py \
  --sim_idx 0 \
  --layout triple \
  --output_video outputs/videos/shear_flow_triple.mp4

# 仅专注于超清涡量场大屏演化
python scripts/generate_shear_flow_video.py \
  --layout vorticity \
  --output_video outputs/videos/shear_flow_vorticity_focus.mp4
```
