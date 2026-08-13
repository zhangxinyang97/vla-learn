---
title: 分层架构-VLM加技能控制器
tags: [layered-architecture, vlm, imitation-learning, diffusion-policy, humanplus, susie, atm, manipgen, dexgraspvla]
aliases: [HumanPlus, Susie, ATM, ManipGen, DexGraspVLA]
---

# 分层架构（二）：High-level Planner + Low-level Controller

> VLM + 模仿/强化学习的 VA 控制器或 Diffusion Policy。任务与技能执行均用学习策略。代表性：HumanPlus / Susie / ATM / SPOT / ManipGen / DexGraspVLA。

## HumanPlus (Stanford)
- 分层解耦任务学习 + 运动控制
- **HST**（Humanoid Shadowing Transformer）：RL 仿真 + 40h 人类运动数据训练低级策略，RGB 实时「影随」人类全身/手部
- **HIT**（Humanoid Imitation Transformer）：行为克隆，自中心视觉高层技能策略，自主完成任务

## Susie (Sergey Levine)
- 任务规划 = **image-editing**：Diffusion 生成任务指令引导的 image subgoal
- goal-conditioned 模仿学习策略 **GCBC** 执行 subgoal-reaching
- 闭环：generate → execute → replan 循环
- 轻量松耦合，可享第三方视频生成模型迭代红利

## ATM (Any-point Trajectory Modeling, 高阳, RSS 2024)
- 预训练轨迹模型预测视频帧内**任意点未来轨迹**（等效光流）
- 点捕获物体空间移动归纳偏置，分离运动与色彩纹理 → 跨具身一致性匹配
- 利用无动作标签人类视频 → 小样本动作标签学鲁棒策略
- RSS 2024 全数审稿人满分

## SPOT (NVIDIA, 小插曲)
- SE(3) Pose Trajectory Diffusion，以对象为中心的扩散策略
- 对象位姿跟踪（object pose tracking）→ Diffusion 预测目标路径 → 任务空间轨迹跟踪
- 与 ATM 区别：SE(3) 描述（非像素点），对运动实现更友好（可接 WBC/轨迹优化）
- 非完整 VL-A 框架，可套 OmniManip 壳模块化替换

## ManipGen (Local Policies)
- **局部策略**：感知+执行限制在操作对象局部周边
- 预训练 VLM 分解长序列任务 → MotionPlanning 长距运动 → 局部范围调 RL+多任务蒸馏的 generalist 技能 VA
- 位置/技能顺序/场景配置不变性；50 项真实任务超越 SayCan/OpenVLA/VoxPoser 等

## DexGraspVLA (北大-灵初)
- 分层 VLA（命名像端到端，实为分层）：VLM 高层规划器 + 扩散低级控制器
- 迭代将语言+视觉输入转为**域不变表示**，缓解域转移
- 视觉分割检测 → **bounding-box 几何化掩码**引导低层，弱化语言/图像域差异对控制的影响
- 零样本面对数千未见物体/光照/背景组合，成功率 >90%

## π0 归位澄清
π0 虽 VLM+FlowMatching 看似分层，但训练**非分层解耦**（VLM 扩展成 VLA 后全量微调，一次性贯通），推理端到端 → **正编端到端 VLA**。
> 端到端 ≠ 具体模型实现，而看 Pipeline 训练/推理逻辑通路。

> Chelsea Finn 观点：只搞大脑 = 非具身智能；**加入小脑（运动控制闭环）才构成具身智能**。VLA 重点在搞「小脑」。

[[端到端VLA模型盘点-下]] | [[Diffusion-Policy详解]] | [[ACT算法详解]] | [[强化学习策略-PPO与SAC]]
