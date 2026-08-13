---
title: 从VLM到空间智能
tags: [spatial-intelligence, 3dgs, gaussian-splatting, sim2real, cast, robogsim, 3d-reconstruction]
aliases: [VLM到空间智能, RoboGSim, CAST, 2D生成3D]
---

# 从 VLM 到空间智能

## VLM 到 3D 空间的 Gap
- 分层「VLM + MotionPlan」架构中，Motion Planning 有成熟最优控制兜底，只差 VLM 解决高维轨迹搜索
- 瓶颈：**2D 图像 grounding 生成的 3D 空间约束映射不准确** → 任务成功率不高（ReKep/OmniManip 共性）
- 重点：VLM 视觉语言模态 与 传统 MP 所需 3D 空间约束 的精细化对齐 = **空间智能**

## 由 2D 生成 3D：3D-GS 是工具
- 3D-GS（Gaussian Splatting）→ 2D 图像 → 3D 场景实时重建
- 海量人类视频 → 3D 结构化/非结构化信息 → 直接供机器人利用（升级版 ATM 意义）
- 省去大量采集：2D-3D 对齐后可从视频合成新数据
- 以对象为中心分层 + **Real2Sim2Real**：只提取末端+被操作主体笛卡尔 3D 轨迹 → 对接 MP，无需引入机器人本体
- 局限：contact-rich / in-hand 操作仍需本体执行机构参与（MP 难建模，用 IL/RL/Diffusion）

## RoboGSim (Real2Sim2Real Gaussian Splatting)
四部分：高斯重建器 + 数字孪生生成器 + 场景组合器 + 交互引擎
- 多视角 + MDH 参数 → 3DGS 重建场景/物体 + 机械臂分割 + MDH 运动学驱动图
- 数字孪生：网格重建 + 布局对齐 + 资产互联
- 场景组合器：新物体/场景/视角合成
- 交互引擎：合成图像 + 闭环评估策略；VR/Xbox 采集操作数据

## CAST (Component-Aligned 3D Reconstruction, 单图→高精度3D)
流程四步：
1. **图像解析**：对象级 2D 分割 + 相对深度 + GPT 分析物体空间关系
2. **遮挡感知 3D 生成**：大规模 3D 生成模型独立生成完整几何 + MAE/点云条件，减少遮挡/缺失影响
3. **场景对齐**：计算变换整合网格到点云
4. **物理一致性**：SDF 解决遮挡/穿透/悬浮，符合物理交互规则
（上海科技大学 / 影眸科技 / 华中科技大学）

## 小结
VLM + 空间智能这条路越来越清晰，分层「VLM/空间智能 + Motion Planning」非常 promising。工具会越来越多。

[[3D-Gaussian-Splatting]] | [[DINOv2与自监督视觉特征]] | [[NVIDIA-Sim-to-Real技术]] | [[分层架构-VLM加运动规划]]
