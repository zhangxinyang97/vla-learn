---
title: ChatVLA与训练范式解耦
tags: [vla, training-paradigm, moe, catastrophic-forgetting, task-interference, multi-modal]
aliases: [ChatVLA, 虚假遗忘, 任务干扰, 分阶段对齐训练]
---

# ChatVLA：分阶段对齐训练与训练范式解耦

## 三种训练范式对比（ChatVLA 作者实验）
1. 仅机器人数据训练（VLA 主流，优化控制性能）
2. 思维链(CoT)推理增强机器人数据（辅助信息提升泛化+任务性能）
3. 视觉文本 + 机器人数据联合训练

## 两大结论

### 结论 1：虚假遗忘（Spurious Forgetting）
- 仅机器人数据训练 → 预训练 VLM 组件**灾难性遗忘**（丢失对话/理解能力）
- 实质非完全丢失，而是机器人数据引起的**错位**；用固定推理模板可「重新激活」视觉文本对齐

### 结论 2：任务干扰（Task Interference）
- 加入视觉文本对 → 机器人控制性能**急剧下降**
- 动作生成与理解所需不同表示在共享参数空间竞争 → 需可分离表征学习

## 解决方案：分阶段对齐 + MoE
- **分阶段对齐训练**：掌握初始控制后逐步整合多模态数据
- **MoE**：分离 Action Head 与 LLM Head

## 效果
- MMMU 基准提升 **6×**；MMStar 47.2%；比 ECOT 参数效率更高
- 25 个真实机器人操作任务优于 OpenVLA

## 与 GO-1 ViLLA 的呼应
ChatVLA 的「MoE 分离理解与控制」与 GO-1 的「VLM + Latent Planner + Action Expert」殊途同归 → **理解与控制解耦是 VLA 训练的关键趋势**。

[[端到端VLA模型盘点-下]] | [[VLA大模型]] | [[强化学习策略-PPO与SAC]] | [[策略与生成的统一趋势]]
